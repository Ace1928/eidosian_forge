import os
import unittest
import yaml
from gae_ext_runtime import testutil
def test_go_custom_runtime_no_write(self):
    """Test generate_config_data with custom runtime."""
    self.write_file('foo.go', 'package main\nfunc main')
    cfg_files = self.generate_config_data(custom=True)
    self.assert_file_exists_with_contents('app.yaml', 'env: flex\nruntime: go\n')
    self.assert_genfile_exists_with_contents(cfg_files, 'Dockerfile', self.read_runtime_def_file('data', 'Dockerfile'))
    self.assert_genfile_exists_with_contents(cfg_files, '.dockerignore', self.read_runtime_def_file('data', 'dockerignore'))