import collections
from unittest import mock
from oslo_vmware import dvs_util
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_get_vlan_spec(self):
    session = mock.Mock()
    spec = dvs_util.get_vlan_spec(session, 7)
    self.assertEqual(7, spec.vlanId)