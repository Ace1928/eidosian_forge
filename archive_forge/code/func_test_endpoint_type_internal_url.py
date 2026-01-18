import os
import sys
import tempfile
from unittest import mock
import uuid
import fixtures
import io
from keystoneauth1 import fixture as keystone_fixture
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from requests_mock.contrib import fixture as rm_fixture
import testscenarios
import testtools
from urllib import parse
from urllib import request
import yaml
from heatclient._i18n import _
from heatclient.common import http
from heatclient.common import utils
from heatclient import exc
import heatclient.shell
from heatclient.tests.unit import fakes
import heatclient.v1.shell
def test_endpoint_type_internal_url(self):
    self.register_keystone_auth_fixture()
    self.useFixture(fixtures.EnvironmentVariable('OS_ENDPOINT_TYPE', 'internalURL'))
    kwargs = {'auth_url': 'http://keystone.example.com:5000/', 'session': mock.ANY, 'auth': mock.ANY, 'service_type': 'orchestration', 'endpoint_type': 'internalURL', 'region_name': '', 'username': 'username', 'password': 'password', 'include_pass': False, 'endpoint_override': mock.ANY}
    heatclient.shell.main(('stack-list',))
    http._construct_http_client.assert_called_once_with(**kwargs)
    heatclient.v1.shell.do_stack_list.assert_called_once()