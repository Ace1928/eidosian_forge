import os
import unittest
import yaml
from gae_ext_runtime import testutil
def test_go_files_in_subdirs(self):
    """Test go runtime does not recognize go files in subdirectories."""
    subdir_path = os.mkdir(os.path.join(self.temp_path, 'subdir'))
    self.write_file(os.path.join('subdir', 'foo.go'), 'package main\nfunc main')
    self.assertEqual(None, self.generate_configs())
    self.assertFalse(os.path.exists(self.full_path('app.yaml')))