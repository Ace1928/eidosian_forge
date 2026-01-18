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
def test_deploy_delete(self):
    self.register_keystone_auth_fixture()
    deploy_resp_dict = {'software_deployment': {'config_id': 'dummy_config_id'}}

    def _get_deployment_request_except(id):
        self.mock_request_error('/software_deployments/%s' % id, 'GET', exc.HTTPNotFound())

    def _delete_deployment_request_except(id):
        self.mock_request_get('/software_deployments/%s' % id, deploy_resp_dict)
        self.mock_request_error('/software_deployments/%s' % id, 'DELETE', exc.HTTPNotFound())

    def _delete_config_request_except(id):
        self.mock_request_get('/software_deployments/%s' % id, deploy_resp_dict)
        self.mock_request_delete('/software_deployments/%s' % id)
        self.mock_request_error('/software_configs/dummy_config_id', 'DELETE', exc.HTTPNotFound())

    def _delete_request_success(id):
        self.mock_request_get('/software_deployments/%s' % id, deploy_resp_dict)
        self.mock_request_delete('/software_deployments/%s' % id)
        self.mock_request_delete('/software_configs/dummy_config_id')
    _get_deployment_request_except('defg')
    _get_deployment_request_except('qwer')
    _delete_deployment_request_except('defg')
    _delete_deployment_request_except('qwer')
    _delete_config_request_except('defg')
    _delete_config_request_except('qwer')
    _delete_request_success('defg')
    _delete_request_success('qwer')
    error = self.assertRaises(exc.CommandError, self.shell, 'deployment-delete defg qwer')
    self.assertIn('Unable to delete 2 of the 2 deployments.', str(error))
    error2 = self.assertRaises(exc.CommandError, self.shell, 'deployment-delete defg qwer')
    self.assertIn('Unable to delete 2 of the 2 deployments.', str(error2))
    output = self.shell('deployment-delete defg qwer')
    self.assertRegex(output, 'Failed to delete the correlative config dummy_config_id of deployment defg')
    self.assertRegex(output, 'Failed to delete the correlative config dummy_config_id of deployment qwer')
    self.assertEqual('', self.shell('deployment-delete defg qwer'))