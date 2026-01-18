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
@mock.patch('builtins.open', new=mock.mock_open(), create=True)
@mock.patch('os.path.exists', return_value=True)
def test_cache_schemas_leaves_auto_switch(self, exists_mock):
    options = {'get_schema': True, 'os_auth_url': self.os_auth_url}
    self.client.schemas.get.return_value = Exception()
    client = mock.MagicMock()
    switch_version = self.shell._cache_schemas(self._make_args(options), client, home_dir=self.cache_dir)
    self.assertEqual(True, switch_version)