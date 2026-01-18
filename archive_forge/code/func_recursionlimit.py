import io
import os.path
import platform
import re
import sys
import traceback
import unittest
from textwrap import dedent
from tempfile import TemporaryDirectory
from IPython.core.ultratb import ColorTB, VerboseTB
from IPython.testing import tools as tt
from IPython.testing.decorators import onlyif_unicode_paths, skip_without
from IPython.utils.syspathcontext import prepended_to_syspath
import sys
def recursionlimit(frames):
    """
    decorator to set the recursion limit temporarily
    """

    def inner(test_function):

        def wrapper(*args, **kwargs):
            rl = sys.getrecursionlimit()
            sys.setrecursionlimit(frames)
            try:
                return test_function(*args, **kwargs)
            finally:
                sys.setrecursionlimit(rl)
        return wrapper
    return inner