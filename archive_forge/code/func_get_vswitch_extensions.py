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
def get_vswitch_extensions(self, vswitch_name):
    vswitch = self._get_vswitch(vswitch_name)
    extensions = self._conn.Msvm_EthernetSwitchExtension(SystemName=vswitch.Name)
    dict_ext_list = [{'name': ext.ElementName, 'version': ext.Version, 'vendor': ext.Vendor, 'description': ext.Description, 'enabled_state': ext.EnabledState, 'extension_type': ext.ExtensionType} for ext in extensions]
    return dict_ext_list