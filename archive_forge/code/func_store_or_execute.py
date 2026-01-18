from logging import error
import os
import sys
from IPython.core.error import TryNext, UsageError
from IPython.core.magic import Magics, magics_class, line_magic
from IPython.lib.clipboard import ClipboardEmpty
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils.text import SList, strip_email_quotes
from IPython.utils import py3compat
def store_or_execute(self, block, name, store_history=False):
    """ Execute a block, or store it in a variable, per the user's request.
        """
    if name:
        self.shell.user_ns[name] = SList(block.splitlines())
        print("Block assigned to '%s'" % name)
    else:
        b = self.preclean_input(block)
        self.shell.user_ns['pasted_block'] = b
        self.shell.using_paste_magics = True
        try:
            self.shell.run_cell(b, store_history)
        finally:
            self.shell.using_paste_magics = False