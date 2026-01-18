import collections
import copy
import functools
import itertools
import math
from oslo_log import log as logging
from heat.common import exception
from heat.common import grouputils
from heat.common.i18n import _
from heat.common import timeutils
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import function
from heat.engine import output
from heat.engine import properties
from heat.engine.resources import stack_resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import support
from heat.scaling import rolling_update
from heat.scaling import template as scl_template
def ignore_param_resolve(snippet, nullable=False):
    if isinstance(snippet, function.Function):
        try:
            result = snippet.result()
        except exception.UserParameterMissing:
            return None
        if not (nullable or function._non_null_value(result)):
            result = None
        return result
    if isinstance(snippet, collections.abc.Mapping):
        return dict(filter(function._non_null_item, ((k, ignore_param_resolve(v, nullable=True)) for k, v in snippet.items())))
    elif not isinstance(snippet, str) and isinstance(snippet, collections.abc.Iterable):
        return list(filter(function._non_null_value, (ignore_param_resolve(v, nullable=True) for v in snippet)))
    return snippet