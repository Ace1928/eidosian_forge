from types import MappingProxyType
from array import array
from frozendict import frozendict
import warnings
from collections.abc import MutableMapping, MutableSequence, MutableSet
def getFreezeTypes():
    return tuple(_freeze_types + [x for x in _freeze_conversion_map_custom] + [x for x in _freeze_conversion_inverse_map_custom])