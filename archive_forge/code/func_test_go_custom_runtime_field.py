import os
import unittest
import yaml
from gae_ext_runtime import testutil
def test_go_custom_runtime_field(self):
    self.write_file('foo.go', 'package main\nfunc main')
    config = testutil.AppInfoFake(runtime='custom', env=2)
    self.assertTrue(self.generate_configs(appinfo=config, deploy=True))