from logging import error
import os
import sys
from IPython.core.error import TryNext, UsageError
from IPython.core.magic import Magics, magics_class, line_magic
from IPython.lib.clipboard import ClipboardEmpty
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils.text import SList, strip_email_quotes
from IPython.utils import py3compat
def preclean_input(self, block):
    lines = block.splitlines()
    while lines and (not lines[0].strip()):
        lines = lines[1:]
    return strip_email_quotes('\n'.join(lines))