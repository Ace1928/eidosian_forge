from unittest import SkipTest
def test_completions_long_style1(self):
    """Suggest corresponding plot options"""
    suggestions = OptsCompleter.line_completer('%%opts AnElement style(', self.completions, self.compositor_defs)
    self.assertEqual(suggestions, ['styleoptA1=', 'styleoptA2='])