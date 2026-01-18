import os
import unittest
import yaml
from gae_ext_runtime import testutil
def assert_no_file(self, filename):
    """Asserts that the relative path 'filename' does not exist."""
    self.assertFalse(os.path.exists(os.path.join(self.temp_path, filename)))