import os
import tempfile
import unittest
from webtest import TestApp
import pecan
from pecan.tests import PecanTestCase
def test_config_with_non_package_relative_import(self):
    from pecan import configuration
    with tempfile.NamedTemporaryFile('wb', suffix='.py') as f:
        f.write(b'\n'.join([b'from . import variables']))
        f.flush()
        configuration.Config({})
        try:
            configuration.conf_from_file(f.name)
        except (ValueError, SystemError, ImportError) as e:
            assert 'relative import' in str(e)
        else:
            raise AssertionError('A relative import-related error should have been raised')