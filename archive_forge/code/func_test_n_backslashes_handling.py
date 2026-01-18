from .. import cmdline, tests
from .features import backslashdir_feature
def test_n_backslashes_handling(self):
    self.requireFeature(backslashdir_feature)
    self.assertAsTokens([(True, '\\\\host\\path')], '"\\\\host\\path"')
    self.assertAsTokens([(False, '\\\\host\\path')], '\\\\host\\path')
    self.assertAsTokens([(True, '\\\\'), (False, '*.py')], '"\\\\\\\\" *.py')
    self.assertAsTokens([(True, '\\\\" *.py')], '"\\\\\\\\\\" *.py"')
    self.assertAsTokens([(True, '\\\\ *.py')], '\\\\\\\\" *.py"')
    self.assertAsTokens([(False, '\\\\"'), (False, '*.py')], '\\\\\\\\\\" *.py')
    self.assertAsTokens([(True, '\\\\')], '"\\\\')