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
def do_jump(self, arg):
    """j(ump) lineno
        Set the next line that will be executed.  Only available in
        the bottom-most frame.  This lets you jump back and execute
        code again, or jump forward to skip code that you don't want
        to run.

        It should be noted that not all jumps are allowed -- for
        instance it is not possible to jump into the middle of a
        for loop or out of a finally clause.
        """
    if self.curindex + 1 != len(self.stack):
        self.error('You can only jump within the bottom frame')
        return
    try:
        arg = int(arg)
    except ValueError:
        self.error("The 'jump' command requires a line number")
    else:
        try:
            self.curframe.f_lineno = arg
            self.stack[self.curindex] = (self.stack[self.curindex][0], arg)
            self.print_stack_entry(self.stack[self.curindex])
        except ValueError as e:
            self.error('Jump failed: %s' % e)