import os
import sys
def setup_to_argv(setup, skip_names=None):
    """
    :param dict setup:
        A dict previously gotten from process_command_line.

    :param set skip_names:
        The names in the setup which shouldn't be converted to argv.

    :note: does not handle --file nor --DEBUG.
    """
    if skip_names is None:
        skip_names = set()
    ret = [get_pydevd_file()]
    for handler in ACCEPTED_ARG_HANDLERS:
        if handler.arg_name in setup and handler.arg_name not in skip_names:
            handler.to_argv(ret, setup)
    return ret