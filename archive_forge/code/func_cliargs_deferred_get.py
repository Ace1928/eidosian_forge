from __future__ import (absolute_import, division, print_function)
from collections.abc import Mapping, Set
from ansible.module_utils.common.collections import is_sequence
from ansible.utils.context_objects import CLIArgs, GlobalCLIArgs
def cliargs_deferred_get(key, default=None, shallowcopy=False):
    """Closure over getting a key from CLIARGS with shallow copy functionality

    Primarily used in ``FieldAttribute`` where we need to defer setting the default
    until after the CLI arguments have been parsed

    This function is not directly bound to ``CliArgs`` so that it works with
    ``CLIARGS`` being replaced
    """

    def inner():
        value = CLIARGS.get(key, default=default)
        if not shallowcopy:
            return value
        elif is_sequence(value):
            return value[:]
        elif isinstance(value, (Mapping, Set)):
            return value.copy()
        return value
    return inner