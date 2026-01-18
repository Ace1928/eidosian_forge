from __future__ import print_function
import os
import fixtures
from pbr.hooks import files
from pbr.tests import base
def test_data_files_globbing(self):
    config = dict(files=dict(data_files='\n  etc/pbr = etc/*'))
    files.FilesConfig(config, 'fake_package').run()
    self.assertIn("\n'etc/pbr/' = \n 'etc/foo'\n'etc/pbr/sub' = \n 'etc/sub/bar'", config['files']['data_files'])