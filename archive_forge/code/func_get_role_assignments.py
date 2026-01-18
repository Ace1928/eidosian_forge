import datetime
import random
import uuid
import freezegun
import http.client
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.resource.backends import base as resource_base
from keystone.tests import unit
from keystone.tests.unit import test_v3
def get_role_assignments(self, expected_status=http.client.OK, **filters):
    """Return the result from querying role assignment API + queried URL.

        Calls GET /v3/role_assignments?<params> and returns its result, where
        <params> is the HTTP query parameters form of effective option plus
        filters, if provided. Queried URL is returned as well.

        :returns: a tuple containing the list role assignments API response and
                  queried URL.

        """
    query_url = self._get_role_assignments_query_url(**filters)
    response = self.get(query_url, expected_status=expected_status)
    return (response, query_url)