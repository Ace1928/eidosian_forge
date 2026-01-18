from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import importlib
import unittest
from unittest import mock
from google.auth import exceptions as google_auth_exceptions
from gslib.command_runner import CommandRunner
from gslib.utils import system_util
import gslib
import gslib.tests.testcase as testcase
@unittest.skipUnless(FIX_GOOGLE_MODULE_FUNCTION_AVAILABLE, 'The gsutil.py file is not available for certain installations like pip.')
@mock.patch.object(importlib, 'reload', autospec=True)
def test_fix_google_module(self, mock_reload):
    with mock.patch.dict('sys.modules', {'google': 'google'}):
        _fix_google_module()
        mock_reload.assert_called_once_with('google')