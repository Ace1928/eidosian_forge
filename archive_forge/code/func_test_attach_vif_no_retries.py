from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import node
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
def test_attach_vif_no_retries(self):
    self.assertIsNone(self.node.attach_vif(self.session, self.vif_id, retry_on_conflict=False))
    self.session.post.assert_called_once_with('nodes/%s/vifs' % self.node.id, json={'id': self.vif_id}, headers=mock.ANY, microversion='1.28', retriable_status_codes=[503])