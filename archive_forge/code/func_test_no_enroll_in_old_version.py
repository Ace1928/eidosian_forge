from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import node
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
def test_no_enroll_in_old_version(self, mock_prov):
    self.node.provision_state = 'enroll'
    self.assertRaises(exceptions.NotSupported, self.node.create, self.session)
    self.assertFalse(self.session.post.called)
    self.assertFalse(mock_prov.called)