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
def test_group_type_create_group_specs(self):
    self._verify('openstack.block_storage.v3.group_type.GroupType.create_group_specs', self.proxy.create_group_type_group_specs, method_args=['value', {'a': 'b'}], expected_args=[self.proxy], expected_kwargs={'specs': {'a': 'b'}})