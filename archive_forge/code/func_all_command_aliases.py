import re
import sys
import breezy
from ... import cmdline, commands, config, help_topics, option, plugin
def all_command_aliases(self):
    for c in self.commands:
        yield from c.aliases