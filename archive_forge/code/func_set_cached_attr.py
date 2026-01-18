import collections
import functools
from oslo_utils import strutils
from heat.common.i18n import _
from heat.engine import constraints as constr
from heat.engine import support
from oslo_log import log as logging
def set_cached_attr(self, key, value):
    self._resolved_values[key] = value
    self._has_new_resolved = True