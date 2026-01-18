from unittest import SkipTest
def test_completions_simple2(self):
    """Same as above even though the selected completion is different"""
    suggestions = OptsCompleter.line_completer('%%opts Anoth', self.completions, self.compositor_defs)
    self.assertEqual(suggestions, self.all_keys)