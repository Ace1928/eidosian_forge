import json
import re
from unittest import mock
from testtools import matchers
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import http_basic
from keystoneauth1 import noauth
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
from keystoneauth1 import token_endpoint
def test_endpoint_data_str(self):
    """Validate EndpointData.__str__."""
    epd = discover.EndpointData(catalog_url='abc', service_type='123', api_version=(2, 3))
    exp = 'EndpointData{api_version=(2, 3), catalog_url=abc, endpoint_id=None, interface=None, major_version=None, max_microversion=None, min_microversion=None, next_min_version=None, not_before=None, raw_endpoint=None, region_name=None, service_id=None, service_name=None, service_type=123, service_url=None, url=abc}'
    self.assertEqual(exp, str(epd))
    self.assertEqual(exp, '%s' % epd)