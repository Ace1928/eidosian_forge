import logging
import mock
import os
import re
import sys
import shutil
import tempfile
import textwrap
import unittest
from gae_ext_runtime import testutil
from gae_ext_runtime import ext_runtime
import constants
def test_java_files_with_war_no_write(self):
    """Test generate_config_data Dockerfile contents with .war file."""
    self.write_file('foo.war', '')
    self.generate_configs()
    self.assert_file_exists_with_contents('app.yaml', self.make_app_yaml('java'))
    self.assert_no_file('Dockerfile')
    self.assert_no_file('.dockerignore')
    cfg_files = self.generate_config_data(deploy=True)
    dockerfile_contents = [constants.DOCKERFILE_JETTY_PREAMBLE, constants.DOCKERFILE_INSTALL_WAR.format('foo.war')]
    self.assert_genfile_exists_with_contents(cfg_files, 'Dockerfile', ''.join(dockerfile_contents))
    self.assert_genfile_exists_with_contents(cfg_files, '.dockerignore', self.read_runtime_def_file('data', 'dockerignore'))