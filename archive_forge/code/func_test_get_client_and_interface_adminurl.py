from unittest import mock
from openstack import config
from ironicclient import client as iroclient
from ironicclient.common import filecache
from ironicclient.common import http
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import client as v1
def test_get_client_and_interface_adminurl(self):
    kwargs = {'project_name': 'PROJECT_NAME', 'username': 'USERNAME', 'password': 'PASSWORD', 'auth_url': 'http://localhost:35357/v2.0', 'service_type': '', 'interface': 'adminURL'}
    self._test_get_client(expected_interface='adminURL', **kwargs)