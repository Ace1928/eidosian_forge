from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_input_redirection(self):
    self._check('foo', None, None, [], ['<foo'])
    self._check('foo', None, None, ['bar'], ['bar', '<foo'])
    self._check('foo', None, None, ['bar'], ['bar', '<', 'foo'])
    self._check('foo', None, None, ['bar'], ['<foo', 'bar'])
    self._check('foo', None, None, ['bar', 'baz'], ['bar', '<foo', 'baz'])