from torch.fx.experimental.graph_gradual_typechecker import Refine
from torch.fx.tensor_type import TensorType
from torch.fx.experimental.unification import Var, unify  # type: ignore[attr-defined]
def infer_symbolic_types(traced):
    """
    Calls our symbolic inferencer twice.
    This is useful when one pass is not enough
    to infer all the information such as the case
    for braodcasting.
    """
    r = Refine(traced)
    r.refine()
    mgu = unify_eq(r.constraints)
    substitute_all_types(traced.graph, mgu)
    r = Refine(traced)
    r.refine()
    mgu = unify_eq(r.constraints)
    substitute_all_types(traced.graph, mgu)
    r.symbolic_relations()