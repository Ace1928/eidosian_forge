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
def test_java_files_with_web_inf_and_yaml_and_open_jdk8_no_env2(self):
    self.write_file('WEB-INF', '')
    config = testutil.AppInfoFake(runtime='java', vm=True, runtime_config=dict(jdk='openjdk8', server='jetty9'))
    self.generate_configs(appinfo=config, deploy=True)
    dockerfile_contents = [constants.DOCKERFILE_COMPAT_PREAMBLE, constants.DOCKERFILE_INSTALL_APP.format('.')]
    self.assert_file_exists_with_contents('Dockerfile', ''.join(dockerfile_contents))
    self.assert_file_exists_with_contents('.dockerignore', self.read_runtime_def_file('data', 'dockerignore'))