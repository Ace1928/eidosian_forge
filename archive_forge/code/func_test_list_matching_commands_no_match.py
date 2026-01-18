import io
import os
import sys
from unittest import mock
from cliff import app as application
from cliff import commandmanager
from cliff import help
from cliff.tests import base
from cliff.tests import utils
def test_list_matching_commands_no_match(self):
    stdout = io.StringIO()
    app = application.App('testing', '1', utils.TestCommandManager(utils.TEST_NAMESPACE), stdout=stdout)
    app.NAME = 'test'
    help_cmd = help.HelpCommand(app, mock.Mock())
    parser = help_cmd.get_parser('test')
    parsed_args = parser.parse_args(['z'])
    self.assertRaises(ValueError, help_cmd.run, parsed_args)