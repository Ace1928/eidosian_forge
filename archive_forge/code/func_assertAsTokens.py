from .. import cmdline, tests
from .features import backslashdir_feature
def assertAsTokens(self, expected, line, single_quotes_allowed=False):
    s = cmdline.Splitter(line, single_quotes_allowed=single_quotes_allowed)
    self.assertEqual(expected, list(s))