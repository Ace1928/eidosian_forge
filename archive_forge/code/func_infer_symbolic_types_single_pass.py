from torch.fx.experimental.graph_gradual_typechecker import Refine
from torch.fx.tensor_type import TensorType
from torch.fx.experimental.unification import Var, unify  # type: ignore[attr-defined]
def infer_symbolic_types_single_pass(traced):
    """
    Calls our symbolic inferencer once.
    """
    r = Refine(traced)
    r.refine()
    mgu = unify_eq(r.constraints)
    substitute_all_types(traced.graph, mgu)