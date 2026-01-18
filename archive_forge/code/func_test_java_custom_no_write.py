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
def test_java_custom_no_write(self):
    """Test generate_config_data with custom=True.

        app.yaml should be written to disk. Also tests correct dockerfile
        contents with a .jar.
        """
    self.write_file('foo.jar', '')
    cfg_files = self.generate_config_data(deploy=False, custom=True)
    self.assert_file_exists_with_contents('app.yaml', self.make_app_yaml('custom'))
    self.assert_genfile_exists_with_contents(cfg_files, '.dockerignore', self.read_runtime_def_file('data', 'dockerignore'))
    dockerfile_contents = [constants.DOCKERFILE_JAVA_PREAMBLE, constants.DOCKERFILE_INSTALL_APP.format('foo.jar'), constants.DOCKERFILE_JAVA8_JAR_CMD.format('foo.jar')]
    self.assert_genfile_exists_with_contents(cfg_files, 'Dockerfile', ''.join(dockerfile_contents))