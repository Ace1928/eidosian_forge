import testscenarios
from unittest import mock
from cliff import command
from cliff import commandmanager
from cliff.tests import base
from cliff.tests import utils
def test_intersected_commands(self):

    def foo(arg):
        pass

    def foo_bar():
        pass
    mgr = utils.TestCommandManager(utils.TEST_NAMESPACE)
    mgr.add_command('foo', foo)
    mgr.add_command('foo bar', foo_bar)
    self.assertIs(foo_bar, mgr.find_command(['foo', 'bar'])[0])
    self.assertIs(foo, mgr.find_command(['foo', 'arg0'])[0])