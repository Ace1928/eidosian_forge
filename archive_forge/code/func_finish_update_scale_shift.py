from typing import Callable, Optional, Tuple, cast
from ..backends import Ops
from ..config import registry
from ..model import Model
from ..types import Floats2d
from ..util import get_width
def finish_update_scale_shift(dY: InT) -> InT:
    model.inc_grad('b', dY.sum(axis=0))
    model.inc_grad('G', (dY * X).sum(axis=0))
    return dY * G