import argparse
import io
import json
import re
import sys
from unittest import mock
import ddt
import fixtures
import keystoneauth1.exceptions as ks_exc
from keystoneauth1.exceptions import DiscoveryFailure
from keystoneauth1.identity.generic.password import Password as ks_password
from keystoneauth1 import session
import requests_mock
from testtools import matchers
import cinderclient
from cinderclient import api_versions
from cinderclient.contrib import noauth
from cinderclient import exceptions
from cinderclient import shell
from cinderclient.tests.unit import fake_actions_module
from cinderclient.tests.unit.fixture_data import keystone_client
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
@requests_mock.Mocker()
def list_volumes_on_service(self, count, mocker):
    os_auth_url = 'http://multiple.service.names/v2.0'
    mocker.register_uri('POST', os_auth_url + '/tokens', text=keystone_client.keystone_request_callback)
    mocker.register_uri('GET', 'http://cinder%i.api.com/' % count, text=json.dumps(fakes.fake_request_get()))
    mocker.register_uri('GET', 'http://cinder%i.api.com/v3/volumes/detail' % count, text='{"volumes": []}')
    self.make_env(include={'OS_AUTH_URL': os_auth_url, 'CINDER_SERVICE_NAME': 'cinder%i' % count})
    _shell = shell.OpenStackCinderShell()
    _shell.main(['list'])