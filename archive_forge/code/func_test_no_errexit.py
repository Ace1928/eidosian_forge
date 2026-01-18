import cmd2
from cliff.interactive import InteractiveApp
from cliff.tests import base
def test_no_errexit(self):
    command_names = set(['show file', 'show folder', 'list all'])
    app = self.make_interactive_app(False, *command_names)
    self.assertFalse(app.errexit)