from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import os
import shutil
import tempfile
import unittest
from gae_ext_runtime import ext_runtime
def assert_file_exists_with_contents(self, filename, contents):
    """Assert that the specified file exists with the given contents.

        Args:
            filename: (str) New file name.
            contents: (str) File contents.
        """
    full_name = self.full_path(filename)
    self.assertTrue(os.path.exists(full_name))
    with open(full_name) as fp:
        actual_contents = fp.read()
    self.assertEqual(contents, actual_contents)