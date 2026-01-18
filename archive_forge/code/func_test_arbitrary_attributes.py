import copy
from unittest import mock
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import session as ksa_session
from openstack.config import cloud_region
from openstack.config import defaults
from openstack import exceptions
from openstack.tests.unit.config import base
from openstack import version as openstack_version
def test_arbitrary_attributes(self):
    cc = cloud_region.CloudRegion('test1', 'region-al', fake_config_dict)
    self.assertEqual('test1', cc.name)
    self.assertEqual('region-al', cc.region_name)
    self.assertEqual('1', cc.a)
    self.assertIsNone(cc.os_b)
    self.assertEqual('3', cc.c)
    self.assertEqual('3', cc.os_c)
    self.assertIsNone(cc.x)
    self.assertFalse(cc.force_ipv4)