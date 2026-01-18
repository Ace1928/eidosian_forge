from types import MappingProxyType
from array import array
from frozendict import frozendict
import warnings
from collections.abc import MutableMapping, MutableSequence, MutableSet
def getFreezeConversionInverseMap():
    return _freeze_conversion_inverse_map | _freeze_conversion_inverse_map_custom