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
def do_unalias(self, arg):
    """unalias name
        Delete the specified alias.
        """
    args = arg.split()
    if len(args) == 0:
        return
    if args[0] in self.aliases:
        del self.aliases[args[0]]