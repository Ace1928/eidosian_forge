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
def defaultFile(self):
    """Produce a reasonable default."""
    filename = self.curframe.f_code.co_filename
    if filename == '<string>' and self.mainpyfile:
        filename = self.mainpyfile
    return filename