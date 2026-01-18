import collections
import enum
import functools
import itertools
import logging
import operator
import sys
from pyomo.common.collections import Sequence, ComponentMap, ComponentSet
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.errors import DeveloperError, InvalidValueError
from pyomo.common.numeric_types import (
from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.base import (
from pyomo.core.base.component import ActiveComponent
from pyomo.core.base.expression import _ExpressionData
from pyomo.core.expr.numvalue import is_fixed, value
import pyomo.core.expr as EXPR
import pyomo.core.kernel as kernel
def ordered_active_constraints(model, config):
    sorter = FileDeterminism_to_SortComponents(config.file_determinism)
    constraints = model.component_data_objects(Constraint, active=True, sort=sorter)
    row_order = config.row_order
    if row_order is None or row_order.__class__ is bool:
        return constraints
    elif isinstance(row_order, ComponentMap):
        row_order = sorted(row_order, key=row_order.__getitem__)
    row_map = {}
    for con in row_order:
        if con.is_indexed():
            for c in con.values(sorter):
                row_map[id(c)] = c
        else:
            row_map[id(con)] = con
    if not row_map:
        return constraints
    row_map = {_id: i for i, _id in enumerate(row_map)}
    _n = len(row_map)
    _row_getter = row_map.get
    return sorted(constraints, key=lambda x: _row_getter(id(x), _n))