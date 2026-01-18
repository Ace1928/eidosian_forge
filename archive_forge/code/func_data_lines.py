import os
from breezy.errors import BinaryFile
from breezy.iterablefile import IterableFile
from breezy.patch import (PatchInvokeError, diff3, iter_patched_from_hunks,
from breezy.patches import parse_patch
from breezy.tests import TestCase, TestCaseInTempDir
def data_lines(self, filename):
    with self.datafile(filename) as datafile:
        return datafile.readlines()