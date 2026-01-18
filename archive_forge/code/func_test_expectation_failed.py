import http.client as http_client
import eventlet.patcher
import httplib2
import webob.dec
import webob.exc
from glance.common import client
from glance.common import exception
from glance.common import wsgi
from glance.tests import functional
from glance.tests import utils
def test_expectation_failed(self):
    """
        Test expectation failed response
        """
    self._do_test_exception('/expectation-failed', exception.UnexpectedStatus)