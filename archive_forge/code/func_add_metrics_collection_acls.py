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
def add_metrics_collection_acls(self, switch_port_name):
    port = self._get_switch_port_allocation(switch_port_name)[0]
    acls = _wqlutils.get_element_associated_class(self._conn, self._PORT_ALLOC_ACL_SET_DATA, element_instance_id=port.InstanceID)
    for acl_type in [self._ACL_TYPE_IPV4, self._ACL_TYPE_IPV6]:
        for acl_dir in [self._ACL_DIR_IN, self._ACL_DIR_OUT]:
            _acls = self._filter_acls(acls, self._ACL_ACTION_METER, acl_dir, acl_type)
            if not _acls:
                acl = self._create_acl(acl_dir, acl_type, self._ACL_ACTION_METER)
                self._jobutils.add_virt_feature(acl, port)