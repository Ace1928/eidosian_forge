import sys
import types
import logging
from weakref import ref as weakref_ref
from pyomo.common.pyomo_typing import overload
from pyomo.common.autoslots import AutoSlots
from pyomo.common.deprecation import deprecation_warning, RenamedClass
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NOTSET
from pyomo.common.numeric_types import native_types, value as expr_value
from pyomo.common.timing import ConstructionTimer
from pyomo.core.expr.numvalue import NumericValue
from pyomo.core.base.component import ComponentData, ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.indexed_component import (
from pyomo.core.base.initializer import Initializer
from pyomo.core.base.misc import apply_indexed_rule, apply_parameterized_indexed_rule
from pyomo.core.base.set import Reals, _AnySet, SetInitializer
from pyomo.core.base.units_container import units
from pyomo.core.expr import GetItemExpression
def store_values(self, new_values, check=True):
    """
        A utility to update a Param with a dictionary or scalar.

        If check=True, then both the index and value
        are checked through the __getitem__ method.  Using check=False
        should only be used by developers!
        """
    if not self._mutable:
        _raise_modifying_immutable_error(self, '*')
    _srcType = type(new_values)
    _isDict = _srcType is dict or (hasattr(_srcType, '__getitem__') and (not isinstance(new_values, NumericValue)))
    if check:
        if _isDict:
            for index, new_value in new_values.items():
                self[index] = new_value
        else:
            for index in self._index_set:
                self[index] = new_values
        return
    if self.is_indexed():
        if _isDict:
            for index, new_value in new_values.items():
                if index not in self._data:
                    self._data[index] = _ParamData(self)
                self._data[index]._value = new_value
        elif not self._data:
            for index in self._index_set:
                p = self._data[index] = _ParamData(self)
                p._value = new_values
        elif len(self._data) == len(self._index_set):
            for index in self._index_set:
                self._data[index]._value = new_values
        else:
            for index in self._index_set:
                if index not in self._data:
                    self._data[index] = _ParamData(self)
                self._data[index]._value = new_values
    else:
        if _isDict:
            if None not in new_values:
                raise RuntimeError('Cannot store value for scalar Param %s:\n\tNo value with index None in the new values dict.' % (self.name,))
            new_values = new_values[None]
        self[None] = new_values