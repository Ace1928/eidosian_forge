from typing import Dict, List, MutableMapping, Optional, Set, Tuple
from onnx import GraphProto, ModelProto, TensorProto, checker, helper, utils
def expand_out_dim_graph(graph: GraphProto, dim_idx: int, inplace: Optional[bool]=False) -> GraphProto:
    """Inserts an extra dimension with extent 1 to each output in the graph.

    Inserts an Unsqueeze node for each output. It can be used as a utility before merging graphs,
    for example when the second one expects a batch dimension.

    Arguments:
        graph (GraphProto): Graph
        dim_idx (int): Index of the dimension to be inserted.
                       A negative value means counting dimensions from the back.
        inplace (bool): If True, mutates the model directly.
                        Otherwise, a copy will be created

    Returns:
        GraphProto
    """
    if type(graph) is not GraphProto:
        raise ValueError('graph argument is not an ONNX graph')
    if not inplace:
        g = GraphProto()
        g.CopyFrom(graph)
    else:
        g = graph
    orig_out_names = [output.name for output in g.output]
    for n in g.node:
        for i, out in enumerate(n.output):
            if out in orig_out_names:
                n.output[i] = out + f'_collapsed_dim_{dim_idx}'
        for i, inp in enumerate(n.input):
            if inp in orig_out_names:
                n.input[i] = inp + f'_collapsed_dim_{dim_idx}'
    expand_dim_k = g.name + '_expand_out_dim_idx'
    g.node.append(helper.make_node('Constant', inputs=[], outputs=[expand_dim_k], name=f'{expand_dim_k}-constant', value=helper.make_tensor(name=f'{expand_dim_k}-value', data_type=TensorProto.INT64, dims=[1], vals=[dim_idx])))
    for _ in range(len(g.output)):
        o = g.output.pop(0)
        prev_output = o.name + f'_collapsed_dim_{dim_idx}'
        g.node.append(helper.make_node('Unsqueeze', inputs=[prev_output, expand_dim_k], outputs=[o.name], name=f'unsqueeze-{o.name}'))
        new_shape = [d.dim_value for d in o.type.tensor_type.shape.dim]
        new_shape.insert(dim_idx, 1)
        g.output.append(helper.make_tensor_value_info(o.name, o.type.tensor_type.elem_type, new_shape))
    return g