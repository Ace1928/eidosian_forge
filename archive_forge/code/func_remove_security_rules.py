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
def remove_security_rules(self, switch_port_name, sg_rules):
    port = self._get_switch_port_allocation(switch_port_name)[0]
    acls = _wqlutils.get_element_associated_class(self._conn, self._PORT_EXT_ACL_SET_DATA, element_instance_id=port.InstanceID)
    remove_acls = []
    for sg_rule in sg_rules:
        filtered_acls = self._filter_security_acls(sg_rule, acls)
        remove_acls.extend(filtered_acls)
    if remove_acls:
        self._jobutils.remove_multiple_virt_features(remove_acls)
        new_acls = [a for a in acls if a not in remove_acls]
        self._sg_acl_sds[port.ElementName] = new_acls