from typing import Sequence, Tuple, TypeVar, Union
from ..model import Model
from ..types import ArrayXd, FloatsXd, IntsXd
def ints_getitem(index: Index) -> Model[IntsXd, IntsXd]:
    """Index into input arrays, and return the subarrays.

    This delegates to `array_getitem`, but allows type declarations.
    """
    return Model('ints-getitem', forward, attrs={'index': index})