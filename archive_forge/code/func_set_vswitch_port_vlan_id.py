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
def set_vswitch_port_vlan_id(self, vlan_id=None, switch_port_name=None, **kwargs):
    """Sets up operation mode, VLAN ID and VLAN trunk for the given port.

        :param vlan_id: the VLAN ID to be set for the given switch port.
        :param switch_port_name: the ElementName of the vSwitch port.
        :param operation_mode: the VLAN operation mode. The acceptable values
            are:
            os_win.constants.VLAN_MODE_ACCESS, os_win.constants.VLAN_TRUNK_MODE
            If not given, VLAN_MODE_ACCESS is used by default.
        :param trunk_vlans: an array of VLAN IDs to be set in trunk mode.
        :raises AttributeError: if an unsupported operation_mode is given, or
            the given operation mode is VLAN_MODE_ACCESS and the given
            trunk_vlans is not None.
        """
    operation_mode = kwargs.get('operation_mode', constants.VLAN_MODE_ACCESS)
    trunk_vlans = kwargs.get('trunk_vlans')
    if operation_mode not in [constants.VLAN_MODE_ACCESS, constants.VLAN_MODE_TRUNK]:
        msg = _('Unsupported VLAN operation mode: %s')
        raise AttributeError(msg % operation_mode)
    if operation_mode == constants.VLAN_MODE_ACCESS and trunk_vlans is not None:
        raise AttributeError(_('The given operation mode is ACCESS, cannot set given trunk_vlans.'))
    port_alloc = self._get_switch_port_allocation(switch_port_name)[0]
    vlan_settings = self._get_vlan_setting_data_from_port_alloc(port_alloc)
    if operation_mode == constants.VLAN_MODE_ACCESS:
        new_vlan_settings = self._prepare_vlan_sd_access_mode(vlan_settings, vlan_id)
    else:
        new_vlan_settings = self._prepare_vlan_sd_trunk_mode(vlan_settings, vlan_id, trunk_vlans)
    if not new_vlan_settings:
        return
    if vlan_settings:
        self._jobutils.remove_virt_feature(vlan_settings)
    self._vlan_sds.pop(port_alloc.InstanceID, None)
    self._jobutils.add_virt_feature(new_vlan_settings, port_alloc)
    vlan_settings = self._get_vlan_setting_data_from_port_alloc(port_alloc)
    if not vlan_settings:
        raise exceptions.HyperVException(_('Port VLAN not found: %s') % switch_port_name)