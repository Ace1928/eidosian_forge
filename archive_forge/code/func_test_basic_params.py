import io
import logging
from testtools import matchers
from keystoneclient import exceptions
from keystoneclient import httpclient
from keystoneclient import session
from keystoneclient.tests.unit import utils
def test_basic_params(self):
    method = 'GET'
    response = 'Test Response'
    status = 200
    self.request(method=method, status_code=status, response=response, headers={'Content-Type': 'application/json'})
    self.assertEqual(self.requests_mock.last_request.method, method)
    logger_message = self.logger_message.getvalue()
    self.assertThat(logger_message, matchers.Contains('curl'))
    self.assertThat(logger_message, matchers.Contains('-X %s' % method))
    self.assertThat(logger_message, matchers.Contains(self.url))
    self.assertThat(logger_message, matchers.Contains(str(status)))
    self.assertThat(logger_message, matchers.Contains(response))