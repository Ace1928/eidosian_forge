import builtins
import copy
import os
import pickle
import contextlib
import subprocess
import socket
import parlai.utils.logging as logging
@contextlib.contextmanager
def override_print(suppress=False, prefix=None):
    """
    Context manager to override the print to suppress or modify output.

    Recommended usage is to call this with suppress=True for all non-primary
    workers, or call with a
    prefix of rank on all workers.

    >>> with override_print(prefix="rank{}".format(rank)):
    ...     my_computation()
    :param bool suppress:
        if true, all future print statements are noops.
    :param str prefix:
        if not None, this string is prefixed to all future print statements.
    """
    builtin_print = builtins.print

    def new_print(*args, **kwargs):
        if suppress:
            return
        elif prefix:
            return builtin_print(prefix, *args, **kwargs)
        else:
            return builtin_print(*args, **kwargs)
    if prefix:
        logging.logger.add_format_prefix(prefix)
    if suppress:
        logging.disable()
    builtins.print = new_print
    yield
    builtins.print = builtin_print
    if suppress:
        logging.enable()