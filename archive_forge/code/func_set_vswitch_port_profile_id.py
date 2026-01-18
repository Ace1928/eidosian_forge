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
def set_vswitch_port_profile_id(self, switch_port_name, profile_id, profile_data, profile_name, vendor_name, **kwargs):
    """Sets up the port profile id.

        :param switch_port_name: The ElementName of the vSwitch port.
        :param profile_id: The profile id to be set for the given switch port.
        :param profile_data: Additional data for the Port Profile.
        :param profile_name: The name of the Port Profile.
        :param net_cfg_instance_id: Unique device identifier of the
            sub-interface.
        :param cdn_label_id: The CDN Label Id.
        :param cdn_label_string: The CDN label string.
        :param vendor_id: The id of the Vendor defining the profile.
        :param vendor_name: The name of the Vendor defining the profile.
        """
    port_alloc = self._get_switch_port_allocation(switch_port_name)[0]
    port_profile = self._get_profile_setting_data_from_port_alloc(port_alloc)
    new_port_profile = self._prepare_profile_sd(profile_id=profile_id, profile_data=profile_data, profile_name=profile_name, vendor_name=vendor_name, **kwargs)
    if port_profile:
        self._jobutils.remove_virt_feature(port_profile)
        self._profile_sds.pop(port_alloc.InstanceID, None)
    try:
        self._jobutils.add_virt_feature(new_port_profile, port_alloc)
    except Exception as ex:
        raise exceptions.HyperVException('Unable to set port profile settings %(port_profile)s for port %(port)s. Error: %(error)s' % dict(port_profile=new_port_profile, port=port_alloc, error=ex))