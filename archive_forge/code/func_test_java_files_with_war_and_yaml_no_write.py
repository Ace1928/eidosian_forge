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
def test_java_files_with_war_and_yaml_no_write(self):
    """Test generate_config_data with .war and fake appinfo."""
    self.write_file('foo.war', '')
    appinfo = testutil.AppInfoFake(runtime='java', env='2', runtime_config=dict(jdk='openjdk8', server='jetty9'))
    cfg_files = self.generate_config_data(appinfo=appinfo, deploy=True)
    dockerfile_contents = [constants.DOCKERFILE_JETTY_PREAMBLE, constants.DOCKERFILE_INSTALL_WAR.format('foo.war')]
    self.assert_genfile_exists_with_contents(cfg_files, 'Dockerfile', ''.join(dockerfile_contents))
    self.assert_genfile_exists_with_contents(cfg_files, '.dockerignore', self.read_runtime_def_file('data', 'dockerignore'))