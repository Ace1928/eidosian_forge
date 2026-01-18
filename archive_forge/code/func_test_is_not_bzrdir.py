from .. import controldir, errors, tests, ui
from .scenarios import load_tests_apply_scenarios
def test_is_not_bzrdir(self):
    self.assertFalse(controldir.is_control_filename('bla'))