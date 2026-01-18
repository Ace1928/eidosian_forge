from abc import abstractmethod
from typing import (
from .config import registry
from .types import Floats2d, Ints1d
from .util import get_array_module, to_categorical
@registry.losses('CosineDistance.v1')
def configure_CosineDistance(*, normalize: bool=True, ignore_zeros: bool=False) -> CosineDistance:
    return CosineDistance(normalize=normalize, ignore_zeros=ignore_zeros)