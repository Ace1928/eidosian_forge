import argparse
import codecs
import io
from unittest import mock
from cliff import app as application
from cliff import command as c_cmd
from cliff import commandmanager
from cliff.tests import base
from cliff.tests import utils as test_utils
from cliff import utils
import sys
def test_list_matching_commands(self):
    stdout = io.StringIO()
    app = application.App('testing', '1', test_utils.TestCommandManager(test_utils.TEST_NAMESPACE), stdout=stdout)
    app.NAME = 'test'
    try:
        self.assertEqual(2, app.run(['t']))
    except SystemExit:
        pass
    output = stdout.getvalue()
    self.assertIn("test: 't' is not a test command. See 'test --help'.", output)
    self.assertIn('Did you mean one of these?', output)
    self.assertIn('three word command\n  two words\n', output)