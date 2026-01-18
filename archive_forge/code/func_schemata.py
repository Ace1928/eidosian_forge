import collections
import functools
from oslo_utils import strutils
from heat.common.i18n import _
from heat.engine import constraints as constr
from heat.engine import support
from oslo_log import log as logging
def schemata(schema):
    """Return dictionary of Schema objects for given dictionary of schemata."""
    return dict(((n, Schema.from_attribute(s)) for n, s in schema.items()))