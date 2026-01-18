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
def test_detach_force(self):
    self._verify('openstack.block_storage.v2.volume.Volume.detach', self.proxy.detach_volume, method_args=['value', '1', True, {'a': 'b'}], expected_args=[self.proxy, '1', True, {'a': 'b'}])