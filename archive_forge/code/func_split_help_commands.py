import os
from ... import commands
from ..test_plugins import BaseTestPlugins
def split_help_commands(self):
    help = {}
    current = None
    out = self.run_bzr_utf8_out('--no-plugins help commands')
    for line in out.splitlines():
        if not line.startswith(' '):
            current = line.split()[0]
        help[current] = help.get(current, '') + line
    return help