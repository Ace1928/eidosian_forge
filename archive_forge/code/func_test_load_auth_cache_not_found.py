import argparse
import copy
import os
from unittest import mock
import fixtures
import testtools
import yaml
from openstack import config
from openstack.config import cloud_region
from openstack.config import defaults
from openstack import exceptions
from openstack.tests.unit.config import base
@mock.patch('openstack.config.cloud_region.keyring')
@mock.patch('keystoneauth1.identity.base.BaseIdentityPlugin.set_auth_state')
def test_load_auth_cache_not_found(self, ks_mock, kr_mock):
    c = config.OpenStackConfig(config_files=[self.cloud_yaml], secure_files=[])
    c._cache_auth = True
    kr_mock.get_password = mock.Mock(side_effect=[RuntimeError])
    region = c.get_one('_test-cloud_')
    kr_mock.get_password.assert_called_with('openstacksdk', region._auth.get_cache_id())
    ks_mock.assert_not_called()