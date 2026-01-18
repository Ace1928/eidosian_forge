from cliff import app as application
from cliff import command
from cliff import commandmanager
from cliff import hooks
from cliff import lister
from cliff import show
from cliff.tests import base
from stevedore import extension
from unittest import mock
def test_get_epilog(self):
    results = self.cmd.get_epilog()
    self.assertIn('hook epilog', results)