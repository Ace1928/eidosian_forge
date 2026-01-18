from typing import Sequence, Tuple, TypeVar, Union
from ..model import Model
from ..types import ArrayXd, FloatsXd, IntsXd
def backprop_get_column(dY):
    dX = model.ops.alloc(shape, dtype=dtype)
    dX[index] = dY
    return dX