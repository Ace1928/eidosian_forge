from unittest import mock
from cliff import app as application
from cliff import commandmanager
from cliff import complete
from cliff.tests import base
def test_complete_command_get_actions(self):
    sot, app, cmd_mgr = self.given_complete_command()
    app.interactive_mode = False
    actions = sot.get_actions(['complete'])
    self.then_actions_equal(actions)