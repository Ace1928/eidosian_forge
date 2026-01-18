import os
import re
import shlex
import sys
import pygments
from pathlib import Path
from IPython.utils.text import marquee
from IPython.utils import openpy
from IPython.utils import py3compat
def run_cell(self, source):
    """Execute a string with one or more lines of code"""
    self.shell.run_cell(source)