import re
from .. import bzr, commands, controldir, errors, option, registry
from ..builtins import cmd_commit
from ..bzr import knitrepo
from ..commands import parse_args
from . import TestCase
def test_iter_switches(self):
    opt = option.Option('hello', help='fg')
    self.assertEqual(list(opt.iter_switches()), [('hello', None, None, 'fg')])
    opt = option.Option('hello', help='fg', type=int)
    self.assertEqual(list(opt.iter_switches()), [('hello', None, 'ARG', 'fg')])
    opt = option.Option('hello', help='fg', type=int, argname='gar')
    self.assertEqual(list(opt.iter_switches()), [('hello', None, 'GAR', 'fg')])
    registry = controldir.ControlDirFormatRegistry()
    bzr.register_metadir(registry, 'one', 'RepositoryFormat7', 'one help')
    bzr.register_metadir(registry, 'two', 'breezy.bzr.knitrepo.RepositoryFormatKnit1', 'two help')
    registry.set_default('one')
    opt = option.RegistryOption('format', 'format help', registry, value_switches=False)
    self.assertEqual(list(opt.iter_switches()), [('format', None, 'ARG', 'format help')])
    opt = option.RegistryOption('format', 'format help', registry, value_switches=True)
    self.assertEqual(list(opt.iter_switches()), [('format', None, 'ARG', 'format help'), ('default', None, None, 'one help'), ('one', None, None, 'one help'), ('two', None, None, 'two help')])