from cliff import app as application
from cliff import command
from cliff import commandmanager
from cliff import hooks
from cliff import lister
from cliff import show
from cliff.tests import base
from stevedore import extension
from unittest import mock
def test_no_app_or_name(self):
    cmd = TestCommand(None, None)
    self.assertEqual([], cmd._hooks)