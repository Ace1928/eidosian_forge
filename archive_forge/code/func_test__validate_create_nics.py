import base64
import io
import os
import tempfile
from unittest import mock
from novaclient import api_versions
from novaclient import exceptions
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit.fixture_data import floatingips
from novaclient.tests.unit.fixture_data import servers as data
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import servers
def test__validate_create_nics(self):
    if self.cs.api_version > api_versions.APIVersion('2.36'):
        self.assertRaises(ValueError, self.cs.servers._validate_create_nics, None)
    else:
        self.cs.servers._validate_create_nics(None)
        self.assertRaises(ValueError, self.cs.servers._validate_create_nics, mock.Mock())
    self.cs.servers._validate_create_nics(['foo', 'bar'])
    self.cs.servers._validate_create_nics(('foo', 'bar'))