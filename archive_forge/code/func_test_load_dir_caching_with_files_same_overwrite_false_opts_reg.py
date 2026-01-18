import os
from unittest import mock
import yaml
import fixtures
from oslo_config import cfg
from oslo_context import context
from oslo_serialization import jsonutils
from oslotest import base as test_base
from oslo_policy import _cache_handler
from oslo_policy import _checks
from oslo_policy import _parser
from oslo_policy import policy
from oslo_policy.tests import base
def test_load_dir_caching_with_files_same_overwrite_false_opts_reg(self):
    test = getattr(self, 'test_load_directory_caching_with_files_same_but_overwrite_false')
    self._test_scenario_with_opts_registered(test)