import __future__
import difflib
import inspect
import linecache
import os
import pdb
import re
import sys
import traceback
import unittest
from io import StringIO, IncrementalNewlineDecoder
from collections import namedtuple
def get_examples(self, string, name='<string>'):
    """
        Extract all doctest examples from the given string, and return
        them as a list of `Example` objects.  Line numbers are
        0-based, because it's most common in doctests that nothing
        interesting appears on the same line as opening triple-quote,
        and so the first interesting line is called "line 1" then.

        The optional argument `name` is a name identifying this
        string, and is only used for error messages.
        """
    return [x for x in self.parse(string, name) if isinstance(x, Example)]