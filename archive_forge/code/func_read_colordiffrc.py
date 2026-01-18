import re
import sys
from os.path import expanduser
import patiencediff
from . import terminal, trace
from .commands import get_cmd_object
from .patches import (ContextLine, Hunk, HunkLine, InsertLine, RemoveLine,
def read_colordiffrc(path):
    colors = {}
    with open(path) as f:
        for line in f.readlines():
            try:
                key, val = line.split('=')
            except ValueError:
                continue
            key = key.strip()
            val = val.strip()
            tmp = val
            if val.startswith('dark'):
                tmp = val[4:]
            if tmp not in terminal.colors:
                continue
            colors[key] = val
    return colors