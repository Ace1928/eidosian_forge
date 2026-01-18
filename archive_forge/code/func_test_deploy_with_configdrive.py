from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import node
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
def test_deploy_with_configdrive(self):
    result = self.node.set_provision_state(self.session, 'active', config_drive='abcd')
    self.assertIs(result, self.node)
    self.session.put.assert_called_once_with('nodes/%s/states/provision' % self.node.id, json={'target': 'active', 'configdrive': 'abcd'}, headers=mock.ANY, microversion=None, retriable_status_codes=_common.RETRIABLE_STATUS_CODES)