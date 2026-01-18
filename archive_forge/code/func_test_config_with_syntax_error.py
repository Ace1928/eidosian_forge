import os
import tempfile
import unittest
from webtest import TestApp
import pecan
from pecan.tests import PecanTestCase
def test_config_with_syntax_error(self):
    from pecan import configuration
    with tempfile.NamedTemporaryFile('wb') as f:
        f.write(b'\n'.join([b'if false', b'var = 3']))
        f.flush()
        configuration.Config({})
        self.assertRaises(SyntaxError, configuration.conf_from_file, f.name)