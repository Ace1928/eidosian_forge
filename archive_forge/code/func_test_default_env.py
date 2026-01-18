import importlib
import os
from unittest import mock
from osc_lib.tests import utils as osc_lib_test_utils
import wrapt
from openstackclient import shell
def test_default_env(self):
    flag = ''
    kwargs = {'compute_api_version': DEFAULT_COMPUTE_API_VERSION, 'identity_api_version': DEFAULT_IDENTITY_API_VERSION, 'image_api_version': DEFAULT_IMAGE_API_VERSION, 'volume_api_version': DEFAULT_VOLUME_API_VERSION, 'network_api_version': DEFAULT_NETWORK_API_VERSION}
    self._assert_cli(flag, kwargs)