from typing import Dict, List, MutableMapping, Optional, Set, Tuple
from onnx import GraphProto, ModelProto, TensorProto, checker, helper, utils
def add_prefix_graph(graph: GraphProto, prefix: str, rename_nodes: Optional[bool]=True, rename_edges: Optional[bool]=True, rename_inputs: Optional[bool]=True, rename_outputs: Optional[bool]=True, rename_initializers: Optional[bool]=True, rename_value_infos: Optional[bool]=True, inplace: Optional[bool]=False, name_map: Optional[Dict[str, str]]=None) -> GraphProto:
    """Adds a prefix to names of elements in a graph: nodes, edges, inputs, outputs,
    initializers, sparse initializer, value infos.

    It can be used as a utility before merging graphs that have overlapping names.
    Empty names are not prefixed.

    Arguments:
        graph (GraphProto): Graph
        prefix (str): Prefix to be added to each name in the graph
        rename_nodes (bool): Whether to prefix node names
        rename_edges (bool): Whether to prefix node edge names
        rename_inputs (bool): Whether to prefix input names
        rename_outputs (bool): Whether to prefix output names
        rename_initializers (bool): Whether to prefix initializer and sparse initializer names
        rename_value_infos (bool): Whether to prefix value info names
        inplace (bool): If True, mutates the graph directly.
                        Otherwise, a copy will be created
        name_map: (Dict): shared name_map in subgraph

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

    def _prefixed(prefix: str, name: str) -> str:
        return prefix + name if len(name) > 0 else name
    if name_map is None:
        name_map = {}
    if rename_edges:
        for n in g.node:
            for e in n.input:
                name_map[e] = _prefixed(prefix, e)
            for e in n.output:
                name_map[e] = _prefixed(prefix, e)
    if rename_inputs:
        for entry in g.input:
            name_map[entry.name] = _prefixed(prefix, entry.name)
    if rename_outputs:
        for entry in g.output:
            name_map[entry.name] = _prefixed(prefix, entry.name)
    if rename_nodes:
        for n in g.node:
            n.name = _prefixed(prefix, n.name)
            for attribute in n.attribute:
                if attribute.g:
                    add_prefix_graph(attribute.g, prefix, inplace=True, name_map=name_map)
    if rename_initializers:
        for init in g.initializer:
            name_map[init.name] = _prefixed(prefix, init.name)
        for sparse_init in g.sparse_initializer:
            name_map[sparse_init.values.name] = _prefixed(prefix, sparse_init.values.name)
            name_map[sparse_init.indices.name] = _prefixed(prefix, sparse_init.indices.name)
    if rename_value_infos:
        for entry in g.value_info:
            name_map[entry.name] = _prefixed(prefix, entry.name)
    for n in g.node:
        for i, output in enumerate(n.output):
            if n.output[i] in name_map:
                n.output[i] = name_map[output]
        for i, input_ in enumerate(n.input):
            if n.input[i] in name_map:
                n.input[i] = name_map[input_]
    for in_desc in g.input:
        if in_desc.name in name_map:
            in_desc.name = name_map[in_desc.name]
    for out_desc in g.output:
        if out_desc.name in name_map:
            out_desc.name = name_map[out_desc.name]
    for initializer in g.initializer:
        if initializer.name in name_map:
            initializer.name = name_map[initializer.name]
    for sparse_initializer in g.sparse_initializer:
        if sparse_initializer.values.name in name_map:
            sparse_initializer.values.name = name_map[sparse_initializer.values.name]
        if sparse_initializer.indices.name in name_map:
            sparse_initializer.indices.name = name_map[sparse_initializer.indices.name]
    for value_info in g.value_info:
        if value_info.name in name_map:
            value_info.name = name_map[value_info.name]
    return g