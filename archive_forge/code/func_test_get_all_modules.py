from .. import controldir, errors, tests, ui
from .scenarios import load_tests_apply_scenarios
def test_get_all_modules(self):
    format = SampleComponentFormat()
    self.assertEqual(set(), self.registry._get_all_modules())
    self.registry.register(format)
    self.assertEqual({'breezy.tests.test_controldir'}, self.registry._get_all_modules())