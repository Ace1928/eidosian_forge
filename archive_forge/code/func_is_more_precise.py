from torch.fx.experimental.unification import Var  # type: ignore[attr-defined]
from ._compatibility import compatibility
@compatibility(is_backward_compatible=False)
def is_more_precise(t1, t2):
    """
    A binary relation denoted by <= that determines if t1 is more precise than t2.
    The relation is reflexive and transitive.
    returns True if t1 is more precise than t2 and False otherwise.
    Example:
        Dyn >= TensorType((1,2,3))
        int >= Dyn
        int >= int
        TensorType((1,Dyn,3)) <= TensorType((1,2,3))
    """
    if t1 == t2:
        return True
    if isinstance(t2, _DynType):
        return True
    if isinstance(t1, TensorType) and isinstance(t2, TensorType):
        return len(t1.__args__) == len(t2.__args__) and all((is_more_precise(elem1, elem2) for elem1, elem2 in zip(t1.__args__, t2.__args__)))
    else:
        return False