import re
import sys
import breezy
from ... import cmdline, commands, config, help_topics, option, plugin
def global_options(self):
    for name, item in option.Option.OPTIONS.items():
        self.data.global_options.append(('--' + item.name, '-' + item.short_name() if item.short_name() else None, item.help.rstrip()))