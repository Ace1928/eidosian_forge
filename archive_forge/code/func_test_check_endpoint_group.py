import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_check_endpoint_group(self):
    """HEAD /OS-EP-FILTER/endpoint_groups/{endpoint_group_id}.

        Valid endpoint_group_id test case.

        """
    endpoint_group_id = self._create_valid_endpoint_group(self.DEFAULT_ENDPOINT_GROUP_URL, self.DEFAULT_ENDPOINT_GROUP_BODY)
    url = '/OS-EP-FILTER/endpoint_groups/%(endpoint_group_id)s' % {'endpoint_group_id': endpoint_group_id}
    self.head(url, expected_status=http.client.OK)