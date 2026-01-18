import os.path
import sys
import yaml
import time
import logging
from argh import arg, aliases, ArghParser, expects_obj
from wandb_watchdog.version import VERSION_STRING
from wandb_watchdog.utils import load_class
def path_split(pathname_spec, separator=os.path.sep):
    """
    Splits a pathname specification separated by an OS-dependent separator.

    :param pathname_spec:
        The pathname specification.
    :param separator:
        (OS Dependent) `:` on Unix and `;` on Windows or user-specified.
    """
    return list(pathname_spec.split(separator))