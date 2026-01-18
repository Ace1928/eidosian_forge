from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import os
import shutil
import tempfile
import unittest
from gae_ext_runtime import ext_runtime
def read_runtime_def_file(self, *args):
    """Read the entire contents of the file.

        Returns the entire contents of the file identified by a set of
        arguments forming a path relative to the root of the runtime
        definition.

        Args:
            *args: A set of path components (see full_path()).  Note that
                these are relative to the runtime definition root, not the
                temporary directory.
        """
    with open(os.path.join(self.runtime_def_root, *args)) as fp:
        return fp.read()