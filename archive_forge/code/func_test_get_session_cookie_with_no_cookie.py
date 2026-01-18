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
def test_get_session_cookie_with_no_cookie(self):
    svc_obj = service.Service()
    cookie = mock.Mock()
    cookie.name = 'cookie'
    cookie.value = 'xyz'
    svc_obj.client.cookiejar = [cookie]
    self.assertIsNone(svc_obj.get_http_cookie())