import http.client as http
import eventlet.patcher
import webob.dec
import webob.exc
from glance.common import client
from glance.common import exception
from glance.common import wsgi
from glance.tests import functional
from glance.tests import utils
def test_get_without_redirect(self):
    """
        Test GET with no redirect
        """
    response = self.client.do_request('GET', '/')
    self.assertEqual(http.OK, response.status)
    self.assertEqual(b'root', response.read())