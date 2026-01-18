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
def register_keystone_v3_token_fixture(self):
    v3_token = keystone_fixture.V3Token()
    service = v3_token.add_service('orchestration')
    service.add_standard_endpoints(public='http://heat.example.com', admin='http://heat-admin.localdomain', internal='http://heat.localdomain')
    self.requests.post('%s/auth/tokens' % V3_URL, json=v3_token, headers={'X-Subject-Token': self.tokenid})