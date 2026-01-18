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
def test_interactive_mode_cmdloop(self):
    app, command = make_app()
    app.interactive_app_factory = mock.MagicMock(name='interactive_app_factory')
    self.assertIsNone(app.interpreter)
    ret = app.run([])
    self.assertIsNotNone(app.interpreter)
    cmdloop = app.interactive_app_factory.return_value.cmdloop
    cmdloop.assert_called_once_with()
    self.assertNotEqual(ret, 0)