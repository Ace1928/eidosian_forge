from unittest import mock
import uuid
from openstack.identity.v3 import endpoint
from openstack.identity.v3 import limit as klimit
from openstack.identity.v3 import registered_limit
from oslo_config import cfg
from oslo_config import fixture as config_fixture
from oslotest import base
from oslo_limit import exception
from oslo_limit import fixture
from oslo_limit import limit
from oslo_limit import opts
def test_get_limit_cache(self, cache=True):
    fix = self.useFixture(fixture.LimitFixture({'foo': 5}, {}))
    project_id = uuid.uuid4().hex
    utils = limit._EnforcerUtils(cache=cache)
    foo_limit = utils._get_limit(project_id, 'foo')
    self.assertEqual(5, foo_limit)
    self.assertEqual(1, fix.mock_conn.registered_limits.call_count)
    foo_limit = utils._get_limit(project_id, 'foo')
    self.assertEqual(5, foo_limit)
    count = 1 if cache else 2
    self.assertEqual(count, fix.mock_conn.registered_limits.call_count)
    fix.projlimits[project_id] = {'foo': 1}
    foo_limit = utils._get_limit(project_id, 'foo')
    self.assertEqual(1, foo_limit)
    self.assertEqual(3, fix.mock_conn.limits.call_count)
    foo_limit = utils._get_limit(project_id, 'foo')
    self.assertEqual(1, foo_limit)
    count = 3 if cache else 4
    self.assertEqual(count, fix.mock_conn.limits.call_count)