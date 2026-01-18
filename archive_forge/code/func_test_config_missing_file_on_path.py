import os
import tempfile
import unittest
from webtest import TestApp
import pecan
from pecan.tests import PecanTestCase
def test_config_missing_file_on_path(self):
    from pecan import configuration
    path = ('bad', 'bad', 'doesnotexist.py')
    configuration.Config({})
    self.assertRaises(RuntimeError, configuration.conf_from_file, os.path.join(__here__, 'config_fixtures', *path))