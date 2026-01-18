from .. import cmdline, tests
from .features import backslashdir_feature
def test_multiple_quoted_args(self):
    self.assertAsTokens([(True, 'x x'), (True, 'y y')], '"x x" "y y"')
    self.assertAsTokens([(True, 'x x'), (True, 'y y')], '"x x" \'y y\'', single_quotes_allowed=True)