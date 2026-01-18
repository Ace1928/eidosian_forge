from __future__ import print_function
import os
import fixtures
from pbr.hooks import files
from pbr.tests import base
def test_auto_package(self):
    config = dict(files=dict(packages='fake_package'))
    files.FilesConfig(config, 'fake_package').run()
    self.assertIn('subpackage', config['files']['packages'])