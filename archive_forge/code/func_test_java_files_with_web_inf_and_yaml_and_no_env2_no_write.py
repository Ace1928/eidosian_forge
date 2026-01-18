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
def test_java_files_with_web_inf_and_yaml_and_no_env2_no_write(self):
    """Test generate_config_data with .war, fake appinfo, env != 2."""
    self.write_file('WEB-INF', '')
    config = testutil.AppInfoFake(runtime='java', vm=True, runtime_config=dict(server='jetty9'))
    cfg_files = self.generate_config_data(appinfo=config, deploy=True)
    dockerfile_contents = [constants.DOCKERFILE_LEGACY_PREAMBLE, constants.DOCKERFILE_INSTALL_APP.format('.')]
    self.assert_genfile_exists_with_contents(cfg_files, 'Dockerfile', ''.join(dockerfile_contents))
    self.assert_genfile_exists_with_contents(cfg_files, '.dockerignore', self.read_runtime_def_file('data', 'dockerignore'))