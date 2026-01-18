from .. import debug, errors, lock, tests
from .scenarios import load_tests_apply_scenarios
def test_create_read_lock(self):
    r_lock = self.read_lock('a-lock-file')
    r_lock.unlock()