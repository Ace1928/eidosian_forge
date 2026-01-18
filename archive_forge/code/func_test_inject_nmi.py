from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import node
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
def test_inject_nmi(self):
    self.node.inject_nmi(self.session)
    self.session.put.assert_called_once_with('nodes/%s/management/inject_nmi' % FAKE['uuid'], json={}, headers=mock.ANY, microversion='1.29', retriable_status_codes=_common.RETRIABLE_STATUS_CODES)