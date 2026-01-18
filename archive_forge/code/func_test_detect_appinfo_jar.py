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
def test_detect_appinfo_jar(self):
    self.write_file('foo.jar', '')
    configurator = self.detect()
    self.assertEqual(configurator.generated_appinfo, {'runtime': 'java', 'env': 'flex'})