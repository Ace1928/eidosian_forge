from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
def test_clear_port_sg_acls_cache(self):
    self.netutils._sg_acl_sds[mock.sentinel.port_id] = [mock.sentinel.acl]
    self.netutils.clear_port_sg_acls_cache(mock.sentinel.port_id)
    self.assertNotIn(mock.sentinel.acl, self.netutils._sg_acl_sds)