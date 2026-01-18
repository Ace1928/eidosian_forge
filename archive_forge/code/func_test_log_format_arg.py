import os
from breezy import bedding, tests, workingtree
def test_log_format_arg(self):
    self._make_simple_branch()
    log = self.run_bzr(['log', '--log-format', 'short'])[0]