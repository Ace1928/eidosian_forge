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
def connect_vnic_to_vswitch(self, vswitch_name, switch_port_name):
    port, found = self._get_switch_port_allocation(switch_port_name, create=True, expected=False)
    if found and port.HostResource and port.HostResource[0]:
        return
    vswitch = self._get_vswitch(vswitch_name)
    vnic = self._get_vnic_settings(switch_port_name)
    port.HostResource = [vswitch.path_()]
    port.Parent = vnic.path_()
    if not found:
        vm = self._get_vm_from_res_setting_data(vnic)
        self._jobutils.add_virt_resource(port, vm)
    else:
        self._jobutils.modify_virt_resource(port)