import errno
import os
import select
import socket
import sys
import tempfile
import time
from io import BytesIO
from .. import errors, osutils, tests, trace, win32utils
from . import features, file_utils, test__walkdirs_win32
from .scenarios import load_tests_apply_scenarios
def test_double_set(self):
    """Test that we get the old value out"""
    osutils.set_or_unset_env('BRZ_TEST_ENV_VAR', 'foo')
    old = osutils.set_or_unset_env('BRZ_TEST_ENV_VAR', 'bar')
    self.assertEqual('foo', old)
    self.assertEqual('bar', os.environ.get('BRZ_TEST_ENV_VAR'))