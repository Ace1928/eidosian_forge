import collections
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common.i18n import _
from heat.common import param_utils
from heat.engine import constraints as constr
from heat.engine import function
from heat.engine.hot import parameters as hot_param
from heat.engine import parameters
from heat.engine import support
from heat.engine import translation as trans
def constraint_items(constraint):

    def range_min_max(constraint):
        if constraint.min is not None:
            yield (hot_param.MIN, constraint.min)
        if constraint.max is not None:
            yield (hot_param.MAX, constraint.max)
    if isinstance(constraint, constr.Length):
        yield (hot_param.LENGTH, dict(range_min_max(constraint)))
    elif isinstance(constraint, constr.Range):
        yield (hot_param.RANGE, dict(range_min_max(constraint)))
    elif isinstance(constraint, constr.AllowedValues):
        yield (hot_param.ALLOWED_VALUES, list(constraint.allowed))
    elif isinstance(constraint, constr.AllowedPattern):
        yield (hot_param.ALLOWED_PATTERN, constraint.pattern)