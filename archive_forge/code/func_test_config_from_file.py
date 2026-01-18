import os
import tempfile
import unittest
from webtest import TestApp
import pecan
from pecan.tests import PecanTestCase
def test_config_from_file(self):
    from pecan import configuration
    path = os.path.join(os.path.dirname(__file__), 'config_fixtures', 'config.py')
    configuration.conf_from_file(path)