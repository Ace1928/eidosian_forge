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
def range_min_max(constraint):
    if constraint.min is not None:
        yield (hot_param.MIN, constraint.min)
    if constraint.max is not None:
        yield (hot_param.MAX, constraint.max)