import ast
import os
import re
import sys
import breezy.branch
from breezy import osutils
from breezy.tests import TestCase, TestSkipped, features
def source_file_name(self, package):
    """Return the path of the .py file for package."""
    if getattr(sys, 'frozen', None) is not None:
        raise TestSkipped("can't test sources in frozen distributions.")
    path = package.__file__
    if path[-1] in 'co':
        return path[:-1]
    else:
        return path