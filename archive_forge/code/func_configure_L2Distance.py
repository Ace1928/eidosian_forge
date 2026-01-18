from abc import abstractmethod
from typing import (
from .config import registry
from .types import Floats2d, Ints1d
from .util import get_array_module, to_categorical
@registry.losses('L2Distance.v1')
def configure_L2Distance(*, normalize: bool=True) -> L2Distance:
    return L2Distance(normalize=normalize)