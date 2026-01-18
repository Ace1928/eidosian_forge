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
def do_return(self, arg):
    """r(eturn)
        Continue execution until the current function returns.
        """
    self.set_return(self.curframe)
    return 1