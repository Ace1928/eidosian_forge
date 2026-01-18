import optparse
import re
from typing import Callable, Dict
from . import errors
from . import registry as _mod_registry
from . import revisionspec
@staticmethod
def from_kwargs(name_, help=None, title=None, value_switches=False, enum_switch=True, **kwargs):
    """Convenience method to generate string-map registry options

        name, help, value_switches and enum_switch are passed to the
        RegistryOption constructor.  Any other keyword arguments are treated
        as values for the option, and their value is treated as the help.
        """
    reg = _mod_registry.Registry()
    for name, switch_help in sorted(kwargs.items()):
        name = name.replace('_', '-')
        reg.register(name, name, help=switch_help)
        if not value_switches:
            help = help + '  "' + name + '": ' + switch_help
            if not help.endswith('.'):
                help = help + '.'
    return RegistryOption(name_, help, reg, title=title, value_switches=value_switches, enum_switch=enum_switch)