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
def lineinfo(self, identifier):
    failed = (None, None, None)
    idstring = identifier.split("'")
    if len(idstring) == 1:
        id = idstring[0].strip()
    elif len(idstring) == 3:
        id = idstring[1].strip()
    else:
        return failed
    if id == '':
        return failed
    parts = id.split('.')
    if parts[0] == 'self':
        del parts[0]
        if len(parts) == 0:
            return failed
    fname = self.defaultFile()
    if len(parts) == 1:
        item = parts[0]
    else:
        f = self.lookupmodule(parts[0])
        if f:
            fname = f
        item = parts[1]
    answer = find_function(item, fname)
    return answer or failed