from .. import cmdline, tests
from .features import backslashdir_feature
def test_empty_result(self):
    self.assertAsTokens([], '')
    self.assertAsTokens([], '    ')