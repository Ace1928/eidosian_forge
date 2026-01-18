from typing import Callable, Optional, TypeVar
from ..config import registry
from ..model import Model
from ..types import Floats2d
def resize_model(model: Model[InT, OutT], new_nO):
    old_layer = model.layers[0]
    new_layer = model.attrs['resize_layer'](old_layer, new_nO)
    model.layers[0] = new_layer
    return model