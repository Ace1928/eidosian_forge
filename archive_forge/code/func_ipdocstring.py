import re
import sys
import unittest
from doctest import DocTestFinder, DocTestRunner, TestResults
from IPython.terminal.interactiveshell import InteractiveShell
def ipdocstring(func):
    """Change the function docstring via ip2py.
    """
    if func.__doc__ is not None:
        func.__doc__ = ip2py(func.__doc__)
    return func