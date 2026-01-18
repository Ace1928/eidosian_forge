import argparse
from collections import OrderedDict
import hashlib
import io
import logging
import os
import sys
import traceback
from unittest import mock
import uuid
import fixtures
from keystoneauth1 import exceptions as ks_exc
from keystoneauth1 import fixture as ks_fixture
from requests_mock.contrib import fixture as rm_fixture
from glanceclient.common import utils
from glanceclient import exc
from glanceclient import shell as openstack_shell
from glanceclient.tests.unit.v2.fixtures import image_show_fixture
from glanceclient.tests.unit.v2.fixtures import image_versions_fixture
from glanceclient.tests import utils as testutils
from glanceclient.v2 import schemas as schemas
import json
@mock.patch('glanceclient.Client')
def test_endpoint_token_no_auth_req(self, mock_client):

    def verify_input(version=None, endpoint=None, *args, **kwargs):
        self.assertIn('token', kwargs)
        self.assertEqual(TOKEN_ID, kwargs['token'])
        self.assertEqual(DEFAULT_IMAGE_URL, endpoint)
        return mock.MagicMock()
    mock_client.side_effect = verify_input
    glance_shell = openstack_shell.OpenStackImagesShell()
    args = ['--os-image-api-version', '2', '--os-auth-token', TOKEN_ID, '--os-image-url', DEFAULT_IMAGE_URL, 'image-list']
    glance_shell.main(args)
    self.assertEqual(1, mock_client.call_count)