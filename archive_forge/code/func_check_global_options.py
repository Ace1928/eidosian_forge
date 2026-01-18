import copy
from .. import Options
def check_global_options(expected_options, white_list=[]):
    """
    returns error message of "" if check Ok
    """
    no_value = object()
    for name, orig_value in expected_options.items():
        if name not in white_list:
            if getattr(Options, name, no_value) != orig_value:
                return 'error in option ' + name
    return ''