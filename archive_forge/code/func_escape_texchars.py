import argparse
import os.path as path
import sys, tempfile, os, re
import logging
import warnings
from subprocess import Popen, PIPE
from . import dotparsing
def escape_texchars(string):
    """Escape the special LaTeX-chars %{}_^

    Examples:

    >>> escape_texchars('10%')
    '10\\\\%'
    >>> escape_texchars('%{}_^\\\\$')
    '\\\\%\\\\{\\\\}\\\\_\\\\^{}$\\\\backslash$\\\\$'
    """
    return ''.join([charmap.get(c, c) for c in string])