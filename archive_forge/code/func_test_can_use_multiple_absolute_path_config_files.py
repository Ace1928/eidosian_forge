import os
from keystone.server.flask import core as server_flask
from keystone.tests import unit
def test_can_use_multiple_absolute_path_config_files(self):
    cfgpaths = [os.path.join(self.custom_config_dir, cfg) for cfg in self.custom_config_files]
    cfgpaths.sort()
    env = {'OS_KEYSTONE_CONFIG_FILES': ';'.join(cfgpaths)}
    config_files = server_flask._get_config_files(env)
    config_files.sort()
    self.assertListEqual(config_files, cfgpaths)
    env = {'OS_KEYSTONE_CONFIG_FILES': ';'.join(cfgpaths + ['', ' '])}
    config_files = server_flask._get_config_files(env)
    config_files.sort()
    self.assertListEqual(config_files, cfgpaths)