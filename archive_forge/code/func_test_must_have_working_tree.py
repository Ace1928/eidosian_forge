from .. import controldir, errors, tests, ui
from .scenarios import load_tests_apply_scenarios
def test_must_have_working_tree(self):
    err = controldir.MustHaveWorkingTree('foo', 'bar')
    self.assertEqual(str(err), "Branching 'bar'(foo) must create a working tree.")