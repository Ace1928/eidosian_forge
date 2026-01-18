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
def complete_undisplay(self, text, line, begidx, endidx):
    return [e for e in self.displaying.get(self.curframe, {}) if e.startswith(text)]