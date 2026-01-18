from .. import cmdline, tests
from .features import backslashdir_feature
def test_ignore_trailing_space(self):
    self.assertAsTokens([(False, 'foo'), (False, 'bar')], 'foo bar  ')