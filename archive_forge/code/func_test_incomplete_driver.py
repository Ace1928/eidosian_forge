import os
from oslotest import base
from requests import HTTPError
import requests_mock
import testtools
from oslo_config import _list_opts
from oslo_config import cfg
from oslo_config import fixture
from oslo_config import sources
from oslo_config.sources import _uri
def test_incomplete_driver(self):
    self.conf_fixture.load_raw_values(group='incomplete_ini_driver', driver='remote_file')
    source = self.conf._open_source_from_opt_group('incomplete_ini_driver')
    self.assertIsNone(source)