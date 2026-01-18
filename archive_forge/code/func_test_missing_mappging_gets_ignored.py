from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import collections
from contextlib import contextmanager
import os
import re
import subprocess
from unittest import mock
from boto import config
from gslib import command
from gslib import command_argument
from gslib import exception
from gslib.commands import rsync
from gslib.commands import version
from gslib.commands import test
from gslib.cs_api_map import ApiSelector
from gslib.tests import testcase
from gslib.utils import boto_util
from gslib.utils import constants
from gslib.utils import shim_util
from gslib.utils import system_util
from gslib.tests import util
def test_missing_mappging_gets_ignored(self):
    with _mock_boto_config({'GSUtil': {'unsupported_field': 'foo'}}):
        flags, env_vars = self._fake_command._translate_boto_config()
        self.assertEqual(flags, [])
        self.assertEqual(env_vars, {})