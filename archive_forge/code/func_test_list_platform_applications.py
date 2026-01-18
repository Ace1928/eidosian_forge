from tests.compat import mock, unittest
from boto.compat import http_client
from boto.sns import connect_to_region
def test_list_platform_applications(self):
    response = self.connection.list_platform_applications()