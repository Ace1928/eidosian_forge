import cmd2
from cliff.interactive import InteractiveApp
from cliff.tests import base
def make_interactive_app(self, errexit, *command_names):
    fake_command_manager = [(x, None) for x in command_names]
    return InteractiveApp(FakeApp, fake_command_manager, stdin=None, stdout=None, errexit=errexit)