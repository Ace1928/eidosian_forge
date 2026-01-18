import os
import unittest
import yaml
from gae_ext_runtime import testutil
def test_go_genfiles_with_go(self):
    """Test generate_config_data with single .go file."""
    self.write_file('foo.go', 'package main\nfunc main')
    self.generate_configs()
    with open(self.full_path('app.yaml')) as f:
        contents = yaml.load(f)
    self.assertEqual(contents, {'runtime': 'go', 'env': 'flex'})
    self.assert_no_file('Dockerfile')
    self.assert_no_file('.dockerignore')
    cfg_files = self.generate_config_data(deploy=True)
    self.assert_genfile_exists_with_contents(cfg_files, 'Dockerfile', self.read_runtime_def_file('data', 'Dockerfile'))
    self.assert_genfile_exists_with_contents(cfg_files, '.dockerignore', self.read_runtime_def_file('data', 'dockerignore'))