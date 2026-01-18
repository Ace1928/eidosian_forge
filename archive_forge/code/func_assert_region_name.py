import copy
from unittest import mock
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import session as ksa_session
from openstack.config import cloud_region
from openstack.config import defaults
from openstack import exceptions
from openstack.tests.unit.config import base
from openstack import version as openstack_version
def assert_region_name(default, compute):
    self.assertEqual(default, cc.region_name)
    self.assertEqual(default, cc.get_region_name())
    self.assertEqual(default, cc.get_region_name(service_type=None))
    self.assertEqual(compute, cc.get_region_name(service_type='compute'))
    self.assertEqual(default, cc.get_region_name(service_type='placement'))