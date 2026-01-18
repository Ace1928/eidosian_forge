from typing import Callable, Tuple
from ..config import registry
from ..model import Model
from ..types import Floats2d
@registry.layers('softmax_activation.v1')
def softmax_activation() -> Model[InT, OutT]:
    return Model('softmax_activation', forward)