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
def set_vswitch_port_sriov(self, switch_port_name, enabled):
    """Enables / Disables SR-IOV for the given port.

        :param switch_port_name: the name of the port which will have SR-IOV
            enabled or disabled.
        :param enabled: boolean, if SR-IOV should be turned on or off.
        """
    self.set_vswitch_port_offload(switch_port_name, sriov_enabled=enabled)