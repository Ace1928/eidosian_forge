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
def extract_values_sparse(self):
    """
        A utility to extract all index-value pairs defined with non-default
        values, returned as a dictionary.

        This method is useful in contexts where key iteration and
        repeated __getitem__ calls are too expensive to extract
        the contents of a parameter.
        """
    if self._mutable:
        ans = {}
        for key, param_value in self.sparse_iteritems():
            ans[key] = param_value()
        return ans
    elif not self.is_indexed():
        return {None: self()}
    else:
        return dict(self.sparse_iteritems())