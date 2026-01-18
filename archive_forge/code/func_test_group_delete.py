from unittest import mock
from openstack.block_storage.v3 import _proxy
from openstack.block_storage.v3 import backup
from openstack.block_storage.v3 import capabilities
from openstack.block_storage.v3 import extension
from openstack.block_storage.v3 import group
from openstack.block_storage.v3 import group_snapshot
from openstack.block_storage.v3 import group_type
from openstack.block_storage.v3 import limits
from openstack.block_storage.v3 import quota_set
from openstack.block_storage.v3 import resource_filter
from openstack.block_storage.v3 import service
from openstack.block_storage.v3 import snapshot
from openstack.block_storage.v3 import stats
from openstack.block_storage.v3 import type
from openstack.block_storage.v3 import volume
from openstack import resource
from openstack.tests.unit import test_proxy_base
def test_group_delete(self):
    self._verify('openstack.block_storage.v3.group.Group.delete', self.proxy.delete_group, method_args=['delete_volumes'], expected_args=[self.proxy], expected_kwargs={'delete_volumes': False})