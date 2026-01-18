from typing import Dict, List, MutableMapping, Optional, Set, Tuple
from onnx import GraphProto, ModelProto, TensorProto, checker, helper, utils
def expand_out_dim(model: ModelProto, dim_idx: int, inplace: Optional[bool]=False) -> ModelProto:
    """Inserts an extra dimension with extent 1 to each output in the graph.

    Inserts an Unsqueeze node for each output. It can be used as a utility before merging graphs,
    for example when the second one expects a batch dimension.

    Arguments:
        model (ModelProto): Model
        dim_idx (int): Index of the dimension to be inserted.
                       A negative value means counting dimensions from the back.
        inplace (bool): If True, mutates the model directly.
                        Otherwise, a copy will be created

    Returns:
        ModelProto
    """
    if type(model) is not ModelProto:
        raise ValueError('model argument is not an ONNX model')
    if not inplace:
        m = ModelProto()
        m.CopyFrom(model)
        model = m
    expand_out_dim_graph(model.graph, dim_idx, inplace=True)
    return model