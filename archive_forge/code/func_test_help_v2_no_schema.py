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
def test_help_v2_no_schema(self):
    shell = openstack_shell.OpenStackImagesShell()
    argstr = '--os-image-api-version 2 help image-create'
    with mock.patch.object(shell, '_get_keystone_auth_plugin') as et_mock:
        actual = shell.main(argstr.split())
        self.assertEqual(0, actual)
        self.assertNotIn('<unavailable>', actual)
        self.assertFalse(et_mock.called)
    argstr = '--os-image-api-version 2 help md-namespace-create'
    with mock.patch.object(shell, '_get_keystone_auth_plugin') as et_mock:
        actual = shell.main(argstr.split())
        self.assertEqual(0, actual)
        self.assertNotIn('<unavailable>', actual)
        self.assertFalse(et_mock.called)
    argstr = '--os-image-api-version 2 help md-resource-type-associate'
    with mock.patch.object(shell, '_get_keystone_auth_plugin') as et_mock:
        actual = shell.main(argstr.split())
        self.assertEqual(0, actual)
        self.assertNotIn('<unavailable>', actual)
        self.assertFalse(et_mock.called)