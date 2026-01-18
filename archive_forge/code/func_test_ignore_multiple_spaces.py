from .. import cmdline, tests
from .features import backslashdir_feature
def test_ignore_multiple_spaces(self):
    self.assertAsTokens([(False, 'foo'), (False, 'bar')], 'foo  bar')