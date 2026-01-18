import collections
from unittest import mock
from oslo_vmware import dvs_util
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_get_port_group_spec(self):
    session = mock.Mock()
    spec = dvs_util.get_port_group_spec(session, 'pg', 7)
    self.assertEqual('pg', spec.name)
    self.assertEqual('ephemeral', spec.type)
    self.assertEqual(7, spec.defaultPortConfig.vlan.vlanId)