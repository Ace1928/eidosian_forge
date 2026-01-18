from unittest import SkipTest
def test_completions_long_plot_closed1(self):
    """Suggest corresponding plot options"""
    suggestions = OptsCompleter.line_completer('%%opts AnElement plot[test=1]', self.completions, self.compositor_defs)
    self.assertEqual(suggestions, self.all_keys)