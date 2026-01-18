from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import node
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
def test_node_add_trait(self):
    self.node.add_trait(self.session, 'CUSTOM_FAKE')
    self.session.put.assert_called_once_with('nodes/%s/traits/%s' % (self.node.id, 'CUSTOM_FAKE'), json=None, headers=mock.ANY, microversion='1.37', retriable_status_codes=_common.RETRIABLE_STATUS_CODES)