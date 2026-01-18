import os
import tempfile
import unittest
from webtest import TestApp
import pecan
from pecan.tests import PecanTestCase
def test_config_with_bad_import(self):
    from pecan import configuration
    path = ('bad', 'importerror.py')
    configuration.Config({})
    self.assertRaises(ImportError, configuration.conf_from_file, os.path.join(__here__, 'config_fixtures', *path))