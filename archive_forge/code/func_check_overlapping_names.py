from typing import Dict, List, MutableMapping, Optional, Set, Tuple
from onnx import GraphProto, ModelProto, TensorProto, checker, helper, utils
def check_overlapping_names(g1: GraphProto, g2: GraphProto, io_map: Optional[List[Tuple[str, str]]]=None) -> List[Tuple[str, List[str]]]:
    """Checks whether there are name collisions between two graphs

    Returns a list of tuples where the first element represents the member containing overlapping names
    (One of: "node", "edge", "value_info", "initializer", "sparse_initializer"), and the
    second element contains a list of names that appear in both graphs on that category.

    Optionally, it takes an io_map, representing the output/inputs to be connected. It provided, overlapping
    present in the io_map argument will be ignored.
    """
    if type(g1) is not GraphProto:
        raise ValueError('g1 argument is not an ONNX graph')
    if type(g2) is not GraphProto:
        raise ValueError('g2 argument is not an ONNX graph')

    def _overlapping(c1: List[str], c2: List[str]) -> List[str]:
        return list(set(c1) & set(c2))

    def _edge_names(graph: GraphProto, exclude: Optional[Set[str]]=None) -> List[str]:
        if exclude is None:
            exclude = set()
        edges = []
        for n in graph.node:
            for i in n.input:
                if i != '' and i not in exclude:
                    edges.append(i)
            for o in n.output:
                if o != '' and o not in exclude:
                    edges.append(o)
        return edges
    result = []
    if not io_map:
        io_map = []
    io_map_inputs = {elem[1] for elem in io_map}
    overlap = _overlapping(_edge_names(g1), _edge_names(g2, exclude=io_map_inputs))
    if overlap:
        result.append(('edge', overlap))
    overlap = _overlapping([e.name for e in g1.value_info], [e.name for e in g2.value_info])
    if overlap:
        result.append(('value_info', overlap))
    overlap = _overlapping([e.name for e in g1.initializer], [e.name for e in g2.initializer])
    if overlap:
        result.append(('initializer', overlap))
    overlap = _overlapping([e.values.name for e in g1.sparse_initializer], [e.values.name for e in g2.sparse_initializer]) + _overlapping([e.indices.name for e in g1.sparse_initializer], [e.indices.name for e in g2.sparse_initializer])
    if overlap:
        result.append(('sparse_initializer', overlap))
    return result