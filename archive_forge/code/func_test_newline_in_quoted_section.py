from .. import cmdline, tests
from .features import backslashdir_feature
def test_newline_in_quoted_section(self):
    self.assertAsTokens([(True, 'foo\nbar\nbaz\n')], '"foo\nbar\nbaz\n"')
    self.assertAsTokens([(True, 'foo\nbar\nbaz\n')], "'foo\nbar\nbaz\n'", single_quotes_allowed=True)