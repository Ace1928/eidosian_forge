import re
from .. import bzr, commands, controldir, errors, option, registry
from ..builtins import cmd_commit
from ..bzr import knitrepo
from ..commands import parse_args
from . import TestCase
def test_conversion(self):
    options = [option.Option('hello')]
    opts, args = self.parse(options, ['--no-hello', '--hello'])
    self.assertEqual(True, opts.hello)
    opts, args = self.parse(options, [])
    self.assertFalse(hasattr(opts, 'hello'))
    opts, args = self.parse(options, ['--hello', '--no-hello'])
    self.assertEqual(False, opts.hello)
    options = [option.Option('number', type=int)]
    opts, args = self.parse(options, ['--number', '6'])
    self.assertEqual(6, opts.number)
    self.assertRaises(errors.CommandError, self.parse, options, ['--number'])
    self.assertRaises(errors.CommandError, self.parse, options, ['--no-number'])
    self.assertRaises(errors.CommandError, self.parse, options, ['--number', 'a'])