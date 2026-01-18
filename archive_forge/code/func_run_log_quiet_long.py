import sys
import unittest
from breezy import tests
def run_log_quiet_long(self, args, env_changes={}):
    cmd = ['--no-aliases', '--no-plugins', '-Oprogress_bar=none', 'log', '-q', '--log-format=long']
    cmd.extend(args)
    return self.run_brz_subprocess(cmd, env_changes=env_changes)