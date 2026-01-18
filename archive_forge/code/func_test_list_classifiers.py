import os
import unittest
import getpass
import urllib
import warnings
from test.support.warnings_helper import check_warnings
from distutils.command import register as register_module
from distutils.command.register import register
from distutils.errors import DistutilsSetupError
from distutils.log import INFO
from distutils.tests.test_config import BasePyPIRCCommandTestCase
def test_list_classifiers(self):
    cmd = self._get_cmd()
    cmd.list_classifiers = 1
    cmd.run()
    results = self.get_logs(INFO)
    self.assertEqual(results, ['running check', 'xxx'])