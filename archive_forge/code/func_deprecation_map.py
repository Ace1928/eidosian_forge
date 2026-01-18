import getpass
import inspect
import os
import sys
import textwrap
import decorator
from magnumclient.common.apiclient import exceptions
from oslo_utils import encodeutils
from oslo_utils import strutils
import prettytable
from magnumclient.i18n import _
def deprecation_map(dep_map):
    """Decorator for applying a map of deprecating arguments to a function.

    The map connects deprecating arguments and their replacements. The
    shell.py script uses this map to create mutually exclusive argument groups
    in argparse and also prints a deprecation warning telling the user to
    switch to the updated argument.

    NOTE: This decorator MUST be the outermost in the chain of argument
    decorators to work correctly.

    Example usage:
    >>> @deprecation_map({ "old-argument": "new-argument" })
    ... @args("old-argument", required=True)
    ... @args("new-argument", required=True)
    ... def do_command_line_stuff():
    ...     pass
    """

    def _decorator(func):
        if not hasattr(func, 'arguments'):
            return func
        func.deprecated_groups = []
        for old_param, new_param in dep_map.items():
            old_info, new_info = (None, None)
            required = False
            for args, kwargs in func.arguments:
                if old_param in args:
                    old_info = (args, kwargs)
                    if 'required' in kwargs:
                        required = kwargs['required']
                    kwargs['required'] = False
                elif new_param in args:
                    new_info = (args, kwargs)
                    kwargs['required'] = False
                if old_info and new_info:
                    break
            func.deprecated_groups.append((old_info, new_info, required))
            func.arguments.remove(old_info)
            func.arguments.remove(new_info)
        return func
    return _decorator