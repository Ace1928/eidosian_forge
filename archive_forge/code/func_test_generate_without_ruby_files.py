import os
import unittest
from gae_ext_runtime import testutil
def test_generate_without_ruby_files(self):
    self.write_file('index.html', 'index')
    self.generate_configs()
    self.assertFalse(os.path.exists(self.full_path('app.yaml')))
    self.assertFalse(os.path.exists(self.full_path('Dockerfile')))
    self.assertFalse(os.path.exists(self.full_path('.dockerignore')))