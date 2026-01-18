import uuid
from keystoneauth1 import exceptions as ks_exc
import requests.exceptions
from openstack.config import cloud_region
from openstack import connection
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def assert_service_disabled(self, service_type, expected_reason, **from_conf_kwargs):
    conn = self._get_conn(**from_conf_kwargs)
    adap = getattr(conn, service_type)
    ex = self.assertRaises(exceptions.ServiceDisabledException, getattr, adap, 'get')
    self.assertIn("Service '%s' is disabled because its configuration could not be loaded." % service_type, ex.message)
    self.assertIn(expected_reason, ex.message)