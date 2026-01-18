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
def set_vswitch_port_mac_spoofing(self, switch_port_name, state):
    """Sets the given port's MAC spoofing to the given state.

        :param switch_port_name: the name of the port which will have MAC
            spoofing set to the given state.
        :param state: boolean, if MAC spoofing should be turned on or off.
        """
    self._set_switch_port_security_settings(switch_port_name, AllowMacSpoofing=state)