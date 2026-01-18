import os
import sys
import simplejson as json
from ..scripts.instance import import_module
def get_command_line_flag(input_spec, is_flag_type=False, input_name=None):
    """
    Generates the command line flag for a given input
    """
    flag, flag_sep = (None, None)
    if input_spec.argstr:
        if '=' in input_spec.argstr:
            if input_spec.argstr.split('=')[1] == '0' or input_spec.argstr.split('=')[1] == '1':
                flag = input_spec.argstr
            else:
                flag = input_spec.argstr.split('=')[0].strip()
                flag_sep = '='
        elif input_spec.argstr.split('%')[0]:
            flag = input_spec.argstr.split('%')[0].strip()
    elif is_flag_type:
        flag = ('--%s' % input_name + ' ').strip()
    return (flag, flag_sep)