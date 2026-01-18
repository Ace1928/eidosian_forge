import re
import sys
import breezy
from ... import cmdline, commands, config, help_topics, option, plugin
def option(self, opt):
    optswitches = {}
    parser = option.get_optparser([opt])
    parser = self.wrap_parser(optswitches, parser)
    optswitches.clear()
    opt.add_option(parser, opt.short_name())
    if isinstance(opt, option.RegistryOption) and opt.enum_switch:
        enum_switch = '--%s' % opt.name
        enum_data = optswitches.get(enum_switch)
        if enum_data:
            try:
                enum_data.registry_keys = opt.registry.keys()
            except ImportError as e:
                enum_data.error_messages.append("ERROR getting registry keys for '--%s': %s" % (opt.name, str(e).split('\n')[0]))
    return sorted(optswitches.values())