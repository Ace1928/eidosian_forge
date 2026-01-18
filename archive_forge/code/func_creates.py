import os
import traceback
import warnings
from os.path import join
from stat import ST_MTIME
import re
import runpy
from docutils import nodes
from docutils.parsers.rst.roles import set_classes
from subprocess import check_call, DEVNULL, CalledProcessError
from pathlib import Path
import matplotlib
def creates():
    """Generator for Python scripts and their output filenames."""
    for dirpath, dirnames, filenames in sorted(os.walk('.')):
        if dirpath.startswith('./build'):
            continue
        for filename in filenames:
            if filename.endswith('.py'):
                path = join(dirpath, filename)
                with open(path) as fd:
                    lines = fd.readlines()
                if len(lines) == 0:
                    continue
                if 'coding: utf-8' in lines[0]:
                    lines.pop(0)
                outnames = []
                for line in lines:
                    if line.startswith('# creates:'):
                        outnames.extend([file.rstrip(',') for file in line.split()[2:]])
                    else:
                        break
                if outnames:
                    yield (dirpath, filename, outnames)