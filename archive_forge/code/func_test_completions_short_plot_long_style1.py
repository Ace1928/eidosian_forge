from unittest import SkipTest
def test_completions_short_plot_long_style1(self):
    """Suggest corresponding plot options"""
    suggestions = OptsCompleter.line_completer('%%opts AnElement [test=1] AnotherElement style(', self.completions, self.compositor_defs)
    self.assertEqual(suggestions, ['styleoptB1=', 'styleoptB2='])