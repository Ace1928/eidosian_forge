from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import node
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
def test_get_console(self):
    self.node.get_console(self.session)
    self.session.get.assert_called_once_with('nodes/%s/states/console' % self.node.id, headers=mock.ANY, microversion=mock.ANY, retriable_status_codes=_common.RETRIABLE_STATUS_CODES)