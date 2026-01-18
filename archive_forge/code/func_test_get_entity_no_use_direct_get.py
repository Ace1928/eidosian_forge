from unittest import mock
from uuid import uuid4
import testtools
from openstack.cloud import _utils
from openstack import exceptions
from openstack.tests.unit import base
def test_get_entity_no_use_direct_get(self):
    uuid = uuid4().hex
    resource = 'network'
    func = 'search_%ss' % resource
    filters = {}
    with mock.patch.object(self.cloud, func) as search:
        _utils._get_entity(self.cloud, resource, uuid, filters)
        search.assert_called_once_with(uuid, filters)