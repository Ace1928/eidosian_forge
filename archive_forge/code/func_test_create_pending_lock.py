from breezy import branch, config, controldir, errors, osutils, tests
from breezy.tests.script import run_script
def test_create_pending_lock(self):
    self.addCleanup(self.config.unlock)
    self.assertTrue(self.config._lock.is_held)