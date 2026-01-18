import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_get_invalid_endpoint_group(self):
    """GET /OS-EP-FILTER/endpoint_groups/{endpoint_group}.

        Invalid endpoint group test case.

        """
    endpoint_group_id = 'foobar'
    url = '/OS-EP-FILTER/endpoint_groups/%(endpoint_group_id)s' % {'endpoint_group_id': endpoint_group_id}
    self.get(url, expected_status=http.client.NOT_FOUND)