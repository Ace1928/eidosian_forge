import os
import io
import re
import sys
import cmd
import bdb
import dis
import code
import glob
import pprint
import signal
import inspect
import tokenize
import functools
import traceback
import linecache
from typing import Union
def interaction(self, frame, traceback):
    if Pdb._previous_sigint_handler:
        try:
            signal.signal(signal.SIGINT, Pdb._previous_sigint_handler)
        except ValueError:
            pass
        else:
            Pdb._previous_sigint_handler = None
    if self.setup(frame, traceback):
        self.forget()
        return
    self.print_stack_entry(self.stack[self.curindex])
    self._cmdloop()
    self.forget()