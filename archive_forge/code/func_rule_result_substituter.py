import inspect
import logging
import sys
import textwrap
import pyomo.core.expr as EXPR
import pyomo.core.base as BASE
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.initializer import Initializer
from pyomo.core.base.component import Component, ActiveComponent
from pyomo.core.base.config import PyomoOptions
from pyomo.core.base.enums import SortComponents
from pyomo.core.base.global_set import UnindexedComponent_set
from pyomo.core.expr.numeric_expr import _ndarray
from pyomo.core.pyomoobject import PyomoObject
from pyomo.common import DeveloperError
from pyomo.common.autoslots import fast_deepcopy
from pyomo.common.collections import ComponentSet
from pyomo.common.deprecation import deprecated, deprecation_warning
from pyomo.common.errors import TemplateExpressionError
from pyomo.common.modeling import NOTSET
from pyomo.common.numeric_types import native_types
from pyomo.common.sorting import sorted_robust
from collections.abc import Sequence
def rule_result_substituter(result_map, map_types):
    _map = result_map
    if map_types is None:
        _map_types = set((type(key) for key in result_map))
    else:
        _map_types = map_types

    def rule_result_substituter_impl(rule, *args, **kwargs):
        if rule.__class__ in _map_types:
            value = rule
        elif isinstance(rule, PyomoObject):
            return rule
        else:
            value = rule(*args, **kwargs)
        if value.__class__ in _map_types and value in _map:
            return _map[value]
        return value
    return rule_result_substituter_impl