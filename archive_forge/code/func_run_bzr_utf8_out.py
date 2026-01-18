import os
from ... import commands
from ..test_plugins import BaseTestPlugins
def run_bzr_utf8_out(self, *args, **kwargs):
    out, _ = self.run_bzr(*args, **kwargs)
    return out