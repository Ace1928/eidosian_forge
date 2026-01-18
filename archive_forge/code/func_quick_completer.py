import glob
import inspect
import os
import re
import sys
from importlib import import_module
from importlib.machinery import all_suffixes
from time import time
from zipimport import zipimporter
from .completer import expand_user, compress_user
from .error import TryNext
from ..utils._process_common import arg_split
from IPython import get_ipython
from typing import List
import_re = re.compile(r'(?P<name>[^\W\d]\w*?)'
def quick_completer(cmd, completions):
    """ Easily create a trivial completer for a command.

    Takes either a list of completions, or all completions in string (that will
    be split on whitespace).

    Example::

        [d:\\ipython]|1> import ipy_completers
        [d:\\ipython]|2> ipy_completers.quick_completer('foo', ['bar','baz'])
        [d:\\ipython]|3> foo b<TAB>
        bar baz
        [d:\\ipython]|3> foo ba
    """
    if isinstance(completions, str):
        completions = completions.split()

    def do_complete(self, event):
        return completions
    get_ipython().set_hook('complete_command', do_complete, str_key=cmd)