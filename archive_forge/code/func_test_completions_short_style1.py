from unittest import SkipTest
def test_completions_short_style1(self):
    """Suggest corresponding plot options"""
    suggestions = OptsCompleter.line_completer('%%opts AnElement (', self.completions, self.compositor_defs)
    self.assertEqual(suggestions, ['styleoptA1=', 'styleoptA2='])