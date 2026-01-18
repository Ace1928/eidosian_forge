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
def do_step(self, arg):
    """s(tep)
        Execute the current line, stop at the first possible occasion
        (either in a function that is called or in the current
        function).
        """
    self.set_step()
    return 1