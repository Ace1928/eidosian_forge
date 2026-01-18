import os
import unittest
from gae_ext_runtime import testutil
def test_custom_runtime(self):
    self.write_file('Dockerfile', 'boring contents')
    self.generate_configs()
    self.assert_file_exists_with_contents('app.yaml', 'env: flex\nruntime: custom\n')