from torch.fx.experimental.graph_gradual_typechecker import Refine
from torch.fx.tensor_type import TensorType
from torch.fx.experimental.unification import Var, unify  # type: ignore[attr-defined]
def substitute_all_types(graph, mapping):
    """
    Apply the most general unifier to all types in a graph
    till reaching a fixed point. If the input and output graph
    are the same, we converge.
    """
    flag = True
    while flag:
        flag = False
        for k in mapping:
            old_mapping_val = mapping[k]
            if mapping[k] in mapping.keys():
                new_key = mapping[k]
                mapping[k] = mapping[new_key]
            if old_mapping_val != mapping[k]:
                flag = True
    for n in graph.nodes:
        n.type = substitute_solution_one_type(mapping, n.type)