from abc import abstractmethod
from typing import (
from .config import registry
from .types import Floats2d, Ints1d
from .util import get_array_module, to_categorical
@registry.losses('SequenceCategoricalCrossentropy.v2')
def configure_SequenceCategoricalCrossentropy_v2(*, normalize: bool=True, names: Optional[Sequence[str]]=None, neg_prefix: Optional[str]=None) -> SequenceCategoricalCrossentropy:
    return SequenceCategoricalCrossentropy(normalize=normalize, names=names, neg_prefix=neg_prefix)