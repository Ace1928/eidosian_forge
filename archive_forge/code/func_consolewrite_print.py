from contextlib import contextmanager
import logging
import typing
import os
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import ffi_proxy
from rpy2.rinterface_lib import conversion
def consolewrite_print(s: str) -> None:
    """R writing to the console/terminal.

    :param s: the data to write to the console/terminal.
    """
    print(s, end='', flush=True)