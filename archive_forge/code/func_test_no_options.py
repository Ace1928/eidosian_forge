import copy
import os
import sys
from unittest import mock
import testtools
from osc_lib import shell
from osc_lib.tests import utils
def test_no_options(self):
    os.environ = {}
    self._assert_initialize_app_arg('', {})
    self._assert_cloud_region_arg('', {})