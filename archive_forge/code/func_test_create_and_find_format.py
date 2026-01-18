from .. import controldir, errors, tests, ui
from .scenarios import load_tests_apply_scenarios
def test_create_and_find_format(self):
    format = NotBzrDirFormat()
    dir = format.initialize(self.get_url())
    self.assertIsInstance(dir, NotBzrDir)
    controldir.ControlDirFormat.register_prober(NotBzrDirProber)
    try:
        found = controldir.ControlDirFormat.find_format(self.get_transport())
        self.assertIsInstance(found, NotBzrDirFormat)
    finally:
        controldir.ControlDirFormat.unregister_prober(NotBzrDirProber)