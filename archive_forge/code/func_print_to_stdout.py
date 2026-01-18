import os
import time
from contextlib import contextmanager
from typing import Callable, Optional
def print_to_stdout(color, str_out):
    """
    The default debug function that prints to standard out.

    :param str color: A string that is an attribute of ``colorama.Fore``.
    """
    col = getattr(Fore, color)
    _lazy_colorama_init()
    print(col + str_out + Fore.RESET)