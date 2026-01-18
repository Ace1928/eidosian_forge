import inspect
import textwrap
import re
import pydoc
from warnings import warn
from collections import namedtuple
from collections.abc import Callable, Mapping
import copy
import sys
def strip_blank_lines(l):
    """Remove leading and trailing blank lines from a list of lines"""
    while l and (not l[0].strip()):
        del l[0]
    while l and (not l[-1].strip()):
        del l[-1]
    return l