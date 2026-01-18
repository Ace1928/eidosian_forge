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
@mock.patch('glanceclient.v2.client.Client')
def test_auth_plugin_invocation_without_username_with_v2(self, v2_client):
    self.make_env(exclude='OS_USERNAME')
    args = '--os-image-api-version 2 image-list'
    glance_shell = openstack_shell.OpenStackImagesShell()
    self.assertRaises(exc.CommandError, glance_shell.main, args.split())