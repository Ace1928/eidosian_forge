import os
from keystone.server.flask import core as server_flask
from keystone.tests import unit
def test_config_files_have_default_values_when_envars_not_set(self):
    config_files = server_flask._get_config_files()
    config_files.sort()
    expected_config_files = []
    self.assertListEqual(config_files, expected_config_files)