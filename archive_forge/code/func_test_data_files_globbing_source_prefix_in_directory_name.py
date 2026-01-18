from __future__ import print_function
import os
import fixtures
from pbr.hooks import files
from pbr.tests import base
def test_data_files_globbing_source_prefix_in_directory_name(self):
    config = dict(files=dict(data_files='\n  share/ansible = ansible/*'))
    files.FilesConfig(config, 'fake_package').run()
    self.assertIn("\n'share/ansible/' = \n'share/ansible/kolla-ansible' = \n'share/ansible/kolla-ansible/test' = \n 'ansible/kolla-ansible/test/baz'", config['files']['data_files'])