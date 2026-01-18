import sys
import types
import eventlet
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import excutils
from heat.common.i18n import _
from heat.common import timeutils
def earlier_than(self, other):
    if other is None:
        return True
    assert isinstance(other, Timeout), 'Invalid type for Timeout compare'
    return self._duration.endtime() < other._duration.endtime()