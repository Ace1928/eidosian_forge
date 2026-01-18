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
def remove_port_qos_rule(self, port_id):
    """Removes the QoS rule from the given port.

        :param port_id: the port's ID from which the QoS rule will be removed.
        """
    port_alloc = self._get_switch_port_allocation(port_id)[0]
    bandwidth = self._get_bandwidth_setting_data_from_port_alloc(port_alloc)
    if bandwidth:
        self._jobutils.remove_virt_feature(bandwidth)
        self._bandwidth_sds.pop(port_alloc.InstanceID, None)