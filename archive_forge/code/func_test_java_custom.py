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
def test_java_custom(self):
    self.write_file('foo.jar', '')
    self.generate_configs(deploy=False, custom=True)
    self.assert_file_exists_with_contents('app.yaml', self.make_app_yaml('custom'))
    self.assert_file_exists_with_contents('.dockerignore', self.read_runtime_def_file('data', 'dockerignore'))
    dockerfile_contents = [constants.DOCKERFILE_JAVA_PREAMBLE, constants.DOCKERFILE_INSTALL_APP.format('foo.jar'), constants.DOCKERFILE_JAVA8_JAR_CMD.format('foo.jar')]
    self.assert_file_exists_with_contents('Dockerfile', ''.join(dockerfile_contents))