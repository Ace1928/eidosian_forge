import io
import re
import sys
from unittest import mock
import fixtures
from keystoneauth1 import fixture
from testtools import matchers
from zunclient import api_versions
from zunclient import exceptions
import zunclient.shell
from zunclient.tests.unit import utils
def test_main_option_region(self):
    self.make_env()
    self._test_main_region('--zun-api-version 1.29 --os-region-name=myregion service-list', 'myregion')