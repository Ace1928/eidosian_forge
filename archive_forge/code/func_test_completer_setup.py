from unittest import SkipTest
def test_completer_setup(self):
    """Test setup_completions for the real completion set"""
    completions = OptsCompleter.setup_completer()
    self.assertEqual(completions, OptsCompleter._completions)
    self.assertNotEqual(completions, {})