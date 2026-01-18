import operator
import os
import shutil
import sys
import textwrap
import tempfile
from unittest import skipIf, TestCase
def isTwistedInstalled():
    try:
        __import__('twisted')
    except ImportError:
        return False
    else:
        return True