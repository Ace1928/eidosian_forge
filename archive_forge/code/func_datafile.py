import os
from breezy.errors import BinaryFile
from breezy.iterablefile import IterableFile
from breezy.patch import (PatchInvokeError, diff3, iter_patched_from_hunks,
from breezy.patches import parse_patch
from breezy.tests import TestCase, TestCaseInTempDir
def datafile(self, filename):
    data_path = os.path.join(os.path.dirname(__file__), 'test_patches_data', filename)
    return open(data_path, 'rb')