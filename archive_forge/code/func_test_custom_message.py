import itertools
import json
import logging
from unittest import mock
from keystoneauth1 import adapter
import requests
from openstack import exceptions
from openstack import format
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
def test_custom_message(self, mock_get_ver):
    mock_get_ver.return_value = '1.1'
    self.assertRaisesRegex(exceptions.NotSupported, 'boom.*1.6 is required, but 1.1 will be used', self.res._assert_microversion_for, self.session, 'fetch', '1.6', error_message='boom')
    mock_get_ver.assert_called_once_with(self.session, action='fetch')