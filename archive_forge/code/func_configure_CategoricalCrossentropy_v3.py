from abc import abstractmethod
from typing import (
from .config import registry
from .types import Floats2d, Ints1d
from .util import get_array_module, to_categorical
@registry.losses('CategoricalCrossentropy.v3')
def configure_CategoricalCrossentropy_v3(*, normalize: bool=True, names: Optional[Sequence[str]]=None, missing_value: Optional[Union[str, int]]=None, neg_prefix: Optional[str]=None, label_smoothing: float=0.0) -> CategoricalCrossentropy:
    return CategoricalCrossentropy(normalize=normalize, names=names, missing_value=missing_value, neg_prefix=neg_prefix, label_smoothing=label_smoothing)