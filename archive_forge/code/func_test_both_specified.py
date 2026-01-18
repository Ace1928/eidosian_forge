import os
import shutil
import tarfile
import tempfile
from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._filesystem import (
def test_both_specified(self):
    self.assertRaises(AssertionError, FileContains, contents=[], matcher=Contains('a'))