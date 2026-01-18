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
def test_java_files_with_war_and_yaml(self):
    self.write_file('foo.war', '')
    appinfo = testutil.AppInfoFake(runtime='java', env='flex', runtime_config=dict(jdk='openjdk8', server='jetty9'))
    self.generate_configs(appinfo=appinfo, deploy=True)
    dockerfile_contents = [constants.DOCKERFILE_JETTY_PREAMBLE, constants.DOCKERFILE_INSTALL_WAR.format('foo.war')]
    self.assert_file_exists_with_contents('Dockerfile', ''.join(dockerfile_contents))
    self.assert_file_exists_with_contents('.dockerignore', self.read_runtime_def_file('data', 'dockerignore'))