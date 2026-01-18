from unittest import mock
from openstack.block_storage.v2 import _proxy
from openstack.block_storage.v2 import backup
from openstack.block_storage.v2 import capabilities
from openstack.block_storage.v2 import limits
from openstack.block_storage.v2 import quota_set
from openstack.block_storage.v2 import snapshot
from openstack.block_storage.v2 import stats
from openstack.block_storage.v2 import type
from openstack.block_storage.v2 import volume
from openstack import resource
from openstack.tests.unit import test_proxy_base
def test_get_defaults(self):
    self._verify('openstack.resource.Resource.fetch', self.proxy.get_quota_set_defaults, method_args=['prj'], expected_args=[self.proxy], expected_kwargs={'error_message': None, 'requires_id': False, 'base_path': '/os-quota-sets/defaults'})