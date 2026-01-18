import http.client
from oslo_serialization import jsonutils
import webtest
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
def assertResponseStatus(self, response, expected_status):
    """Assert a specific status code on the response.

        :param response: :py:class:`httplib.HTTPResponse`
        :param expected_status: The specific ``status`` result expected

        example::

            self.assertResponseStatus(response, http.client.NO_CONTENT)
        """
    self.assertEqual(expected_status, response.status_code, 'Status code %s is not %s, as expected\n\n%s' % (response.status_code, expected_status, response.body))