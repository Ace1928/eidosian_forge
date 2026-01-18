from logging import error
import os
import sys
from IPython.core.error import TryNext, UsageError
from IPython.core.magic import Magics, magics_class, line_magic
from IPython.lib.clipboard import ClipboardEmpty
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils.text import SList, strip_email_quotes
from IPython.utils import py3compat
def rerun_pasted(self, name='pasted_block'):
    """ Rerun a previously pasted command.
        """
    b = self.shell.user_ns.get(name)
    if b is None:
        raise UsageError('No previous pasted block available')
    if not isinstance(b, str):
        raise UsageError("Variable 'pasted_block' is not a string, can't execute")
    print("Re-executing '%s...' (%d chars)" % (b.split('\n', 1)[0], len(b)))
    self.shell.run_cell(b)