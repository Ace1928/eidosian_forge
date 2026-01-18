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
def get_attribute_type(h5_obj: HDF5) -> Type[DatasetAttribute[HDF5, Any, Any]]:
    """
    Returns the ``DatasetAttribute`` of the dataset attribute contained
    in ``h5_obj``.
    """
    type_id = h5_obj.attrs[AttributeInfo.bind_key('type_id')]
    return DatasetAttribute.registry[type_id]