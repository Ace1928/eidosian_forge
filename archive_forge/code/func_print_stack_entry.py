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
def print_stack_entry(self, frame_lineno, prompt_prefix=line_prefix):
    frame, lineno = frame_lineno
    if frame is self.curframe:
        prefix = '> '
    else:
        prefix = '  '
    self.message(prefix + self.format_stack_entry(frame_lineno, prompt_prefix))