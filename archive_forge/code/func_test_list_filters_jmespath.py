import copy
import queue
from unittest import mock
from keystoneauth1 import session
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack import proxy
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
def test_list_filters_jmespath(self):
    fake_response = [FilterableResource(a='a1', b='b1', c='c'), FilterableResource(a='a2', b='b2', c='c'), FilterableResource(a='a3', b='b3', c='c')]
    FilterableResource.list = mock.Mock()
    FilterableResource.list.return_value = fake_response
    rv = self.sot._list(FilterableResource, paginated=False, base_path=None, jmespath_filters="[?c=='c']")
    self.assertEqual(3, len(rv))
    rv = self.sot._list(FilterableResource, paginated=False, base_path=None, jmespath_filters="[?d=='c']")
    self.assertEqual(0, len(rv))