from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import node
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
def test_enroll_new_version(self, mock_prov):
    self.session.default_microversion = '1.11'
    self.node.provision_state = 'enroll'
    self.new_state = 'enroll'
    result = self.node.create(self.session)
    self.assertIs(result, self.node)
    self.session.post.assert_called_once_with(mock.ANY, json={'driver': FAKE['driver']}, headers=mock.ANY, microversion=self.session.default_microversion, params={})
    self.assertFalse(mock_prov.called)