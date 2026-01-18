import collections
import functools
from oslo_utils import strutils
from heat.common.i18n import _
from heat.engine import constraints as constr
from heat.engine import support
from oslo_log import log as logging
def reset_resolved_values(self):
    if hasattr(self, '_resolved_values'):
        self._has_new_resolved = len(self._resolved_values) > 0
    else:
        self._has_new_resolved = False
    self._resolved_values = {}