import functools
import re
from eventlet import patcher
from eventlet import tpool
from oslo_log import log as logging
from oslo_utils import units
import six
from os_win._i18n import _
from os_win import conf
from os_win import constants
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils import baseutils
from os_win.utils import jobutils
def is_metrics_collection_allowed(self, switch_port_name):
    port = self._get_switch_port_allocation(switch_port_name)[0]
    if not self._is_port_vm_started(port):
        return False
    acls = _wqlutils.get_element_associated_class(self._conn, self._PORT_ALLOC_ACL_SET_DATA, element_instance_id=port.InstanceID)
    acls = [a for a in acls if a.Action == self._ACL_ACTION_METER]
    if len(acls) < 2:
        return False
    return True