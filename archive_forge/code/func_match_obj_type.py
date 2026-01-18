import typing
import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping, MutableMapping, Sequence
from functools import lru_cache
from numbers import Number
from types import MappingProxyType
from typing import (
from pennylane.data.base import hdf5
from pennylane.data.base.hdf5 import HDF5, HDF5Any, HDF5Group
from pennylane.data.base.typing_util import UNSET, get_type, get_type_str
def match_obj_type(type_or_obj: Union[ValueType, Type[ValueType]]) -> Type[DatasetAttribute[HDF5Any, ValueType, ValueType]]:
    """
    Returns an ``DatasetAttribute`` that can accept an object of type ``type_or_obj``
    as a value.

    Args:
        type_or_obj: A type or an object

    Returns:
        DatasetAttribute that can accept ``type_or_obj`` (or an object of that
            type) as a value.

    Raises:
        TypeError, if no DatasetAttribute can accept an object of that type
    """
    type_ = get_type(type_or_obj)
    if hasattr(type_, 'type_id'):
        return DatasetAttribute.registry[type_.type_id]
    ret = DatasetAttribute.registry['array']
    if type_ in DatasetAttribute.type_consumer_registry:
        ret = DatasetAttribute.type_consumer_registry[type_]
    elif issubclass(type_, Number):
        ret = DatasetAttribute.registry['scalar']
    elif hasattr(type_, '__array__'):
        ret = DatasetAttribute.registry['array']
    elif issubclass(type_, Sequence):
        ret = DatasetAttribute.registry['list']
    elif issubclass(type_, Mapping):
        ret = DatasetAttribute.registry['dict']
    return ret