from .. import cmdline, tests
from .features import backslashdir_feature
def test_escape_quote(self):
    self.assertAsTokens([(True, 'foo"bar')], '"foo\\"bar"')
    self.assertAsTokens([(True, 'foo\\"bar')], '"foo\\\\\\"bar"')
    self.assertAsTokens([(True, 'foo\\bar')], '"foo\\\\"bar"')