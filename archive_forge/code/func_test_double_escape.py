from .. import cmdline, tests
from .features import backslashdir_feature
def test_double_escape(self):
    self.assertAsTokens([(True, 'foo\\\\bar')], '"foo\\\\bar"')
    self.assertAsTokens([(False, 'foo\\\\bar')], 'foo\\\\bar')