import json
import logging
from unittest import mock
import ddt
import fixtures
from keystoneauth1 import adapter
from keystoneauth1 import exceptions as keystone_exception
from oslo_serialization import jsonutils
from cinderclient import api_versions
import cinderclient.client
from cinderclient import exceptions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
@mock.patch.object(exceptions, 'from_response')
def test_keystone_request_raises_auth_failure_exception(self, mock_from_resp):
    kwargs = {'body': {'volume': {'status': 'creating', 'imageRef': 'username', 'attach_status': 'detached'}, 'authenticated': 'True'}}
    with mock.patch.object(adapter.Adapter, 'request', side_effect=keystone_exception.AuthorizationFailure()):
        session_client = cinderclient.client.SessionClient(session=mock.Mock())
        self.assertRaises(keystone_exception.AuthorizationFailure, session_client.request, mock.sentinel.url, 'POST', **kwargs)
    self.assertFalse(mock_from_resp.called)