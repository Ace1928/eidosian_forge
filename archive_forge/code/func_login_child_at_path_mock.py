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
def login_child_at_path_mock(path):
    if path == 'userName':
        return self.username
    if path == 'password':
        return self.password
    if path == 'sessionID':
        return self.session_id