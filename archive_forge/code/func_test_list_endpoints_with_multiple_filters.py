import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import test_v3
def test_list_endpoints_with_multiple_filters(self):
    """Call ``GET /endpoints?interface={interface}...``.

        Ensure passing different combinations of interface, region_id and
        service_id as filters will return the correct result.

        """
    ref = self._create_random_endpoint(interface='internal')
    response = self.get('/endpoints?interface=%s&region_id=%s' % (ref['interface'], ref['region_id']))
    self.assertValidEndpointListResponse(response, ref=ref)
    for endpoint in response.json['endpoints']:
        self.assertEqual(ref['interface'], endpoint['interface'])
        self.assertEqual(ref['region_id'], endpoint['region_id'])
    ref = self._create_random_endpoint(interface='internal')
    response = self.get('/endpoints?interface=%s&service_id=%s' % (ref['interface'], ref['service_id']))
    self.assertValidEndpointListResponse(response, ref=ref)
    for endpoint in response.json['endpoints']:
        self.assertEqual(ref['interface'], endpoint['interface'])
        self.assertEqual(ref['service_id'], endpoint['service_id'])
    ref = self._create_random_endpoint(interface='internal')
    response = self.get('/endpoints?region_id=%s&service_id=%s' % (ref['region_id'], ref['service_id']))
    self.assertValidEndpointListResponse(response, ref=ref)
    for endpoint in response.json['endpoints']:
        self.assertEqual(ref['region_id'], endpoint['region_id'])
        self.assertEqual(ref['service_id'], endpoint['service_id'])
    ref = self._create_random_endpoint(interface='internal')
    response = self.get('/endpoints?interface=%s&region_id=%s&service_id=%s' % (ref['interface'], ref['region_id'], ref['service_id']))
    self.assertValidEndpointListResponse(response, ref=ref)
    for endpoint in response.json['endpoints']:
        self.assertEqual(ref['interface'], endpoint['interface'])
        self.assertEqual(ref['region_id'], endpoint['region_id'])
        self.assertEqual(ref['service_id'], endpoint['service_id'])