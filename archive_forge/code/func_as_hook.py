import os
import io
import re
import sys
import tempfile
import subprocess
from io import UnsupportedOperation
from pathlib import Path
from IPython import get_ipython
from IPython.display import display
from IPython.core.error import TryNext
from IPython.utils.data import chop
from IPython.utils.process import system
from IPython.utils.terminal import get_terminal_size
from IPython.utils import py3compat
def as_hook(page_func):
    """Wrap a pager func to strip the `self` arg

    so it can be called as a hook.
    """
    return lambda self, *args, **kwargs: page_func(*args, **kwargs)