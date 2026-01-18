import optparse
import re
from typing import Callable, Dict
from . import errors
from . import registry as _mod_registry
from . import revisionspec
def add_option(self, parser, short_name):
    """Add this option to an Optparse parser"""
    if self.value_switches:
        parser = parser.add_option_group(self.title)
    if self.enum_switch:
        Option.add_option(self, parser, short_name)
    if self.value_switches:
        alias_map = self.registry.alias_map()
        for key in self.registry.keys():
            if key in self.registry.aliases():
                continue
            option_strings = ['--%s' % name for name in [key] + [alias for alias in alias_map.get(key, []) if not self.is_hidden(alias)]]
            if self.is_hidden(key):
                help = optparse.SUPPRESS_HELP
            else:
                help = self.registry.get_help(key)
            if self.short_value_switches and key in self.short_value_switches:
                option_strings.append('-%s' % self.short_value_switches[key])
            parser.add_option(*option_strings, action='callback', callback=self._optparse_value_callback(key), help=help)