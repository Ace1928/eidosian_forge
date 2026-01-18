import re
import sys
import breezy
from ... import cmdline, commands, config, help_topics, option, plugin
def command_cases(self):
    cases = ''
    for command in self.data.commands:
        cases += self.command_case(command)
    return cases