import typing
from collections.abc import Mapping
from typing import Dict, Generic, Union
from pennylane.data.base.attribute import DatasetAttribute
from pennylane.data.base.hdf5 import HDF5Any, HDF5Group
from pennylane.data.base.mapper import MapperMixin
from pennylane.data.base.typing_util import T
def copy_value(self) -> Dict[str, T]:
    return {key: attr.copy_value() for key, attr in self._mapper.items()}