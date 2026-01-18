import ast
import os
import re
import sys
import breezy.branch
from breezy import osutils
from breezy.tests import TestCase, TestSkipped, features
def is_our_code(self, fname):
    """True if it's a "real" part of breezy rather than external code"""
    if '/util/' in fname or '/plugins/' in fname:
        return False
    else:
        return True