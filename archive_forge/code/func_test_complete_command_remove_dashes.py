from unittest import mock
from cliff import app as application
from cliff import commandmanager
from cliff import complete
from cliff.tests import base
def test_complete_command_remove_dashes(self):
    sot, app, cmd_mgr = self.given_complete_command()
    parsed_args = mock.Mock()
    parsed_args.name = 'test-take'
    parsed_args.shell = 'bash'
    content = app.stdout.content
    self.assertEqual(0, sot.take_action(parsed_args))
    self.assertIn('_test_take()\n', content[0])
    self.assertIn('complete -F _test_take test-take\n', content[-1])