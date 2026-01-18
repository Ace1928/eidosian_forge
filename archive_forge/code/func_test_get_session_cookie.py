import http.client as httplib
import io
from unittest import mock
import ddt
import requests
import suds
from oslo_vmware import exceptions
from oslo_vmware import service
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_get_session_cookie(self):
    svc_obj = service.Service()
    cookie_value = 'xyz'
    cookie = mock.Mock()
    cookie.name = 'vmware_soap_session'
    cookie.value = cookie_value
    svc_obj.client.cookiejar = [cookie]
    self.assertEqual(cookie_value, svc_obj.get_http_cookie())