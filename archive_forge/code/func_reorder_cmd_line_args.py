import os
import sys
import simplejson as json
from ..scripts.instance import import_module
def reorder_cmd_line_args(cmd_line, interface, ignore_inputs=None):
    """
    Generates a new command line with the positional arguments in the
    correct order
    """
    interface_name = cmd_line.split()[0]
    positional_arg_dict = {}
    positional_args = []
    non_positional_args = []
    for name, spec in sorted(interface.inputs.traits(transient=None).items()):
        if ignore_inputs is not None and name in ignore_inputs:
            continue
        value_key = '[' + name.upper() + ']'
        if spec.position is not None:
            positional_arg_dict[spec.position] = value_key
        else:
            non_positional_args.append(value_key)
    last_arg = None
    for item in sorted(positional_arg_dict.items()):
        if item[0] == -1:
            last_arg = item[1]
            continue
        positional_args.append(item[1])
    return interface_name + ' ' + (' '.join(positional_args) + ' ' if len(positional_args) > 0 else '') + (last_arg + ' ' if last_arg else '') + ' '.join(non_positional_args)