import cmd2
from cliff.interactive import InteractiveApp
from cliff.tests import base
def test_errexit(self):
    command_names = set(['show file', 'show folder', 'list all'])
    app = self.make_interactive_app(True, *command_names)
    self.assertTrue(app.errexit)