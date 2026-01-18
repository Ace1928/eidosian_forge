from __future__ import print_function
import functools
import os
import subprocess
from unittest import TestCase, skipIf
import attr
from .._methodical import MethodicalMachine
from .test_discover import isTwistedInstalled
def isGraphvizInstalled():
    """
    Are the graphviz tools installed?
    """
    r, w = os.pipe()
    os.close(w)
    try:
        return not subprocess.call('dot', stdin=r, shell=True)
    finally:
        os.close(r)