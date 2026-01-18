import os
import re
import shlex
import sys
import pygments
from pathlib import Path
from IPython.utils.text import marquee
from IPython.utils import openpy
from IPython.utils import py3compat
def re_mark(mark):
    return re.compile('^\\s*#\\s+<demo>\\s+%s\\s*$' % mark, re.MULTILINE)