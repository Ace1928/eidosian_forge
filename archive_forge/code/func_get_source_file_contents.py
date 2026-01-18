import ast
import os
import re
import sys
import breezy.branch
from breezy import osutils
from breezy.tests import TestCase, TestSkipped, features
def get_source_file_contents(self, extensions=None):
    for fname in self.get_source_files(extensions=extensions):
        with open(fname) as f:
            yield (fname, f.read())