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
def test_java_custom_runtime_field(self):
    self.write_file('foo.jar', '')
    config = testutil.AppInfoFake(runtime='java', env='2')
    self.assertTrue(self.generate_configs(appinfo=config, deploy=True))