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
def test_java_files_with_too_many_artifacts(self):
    self.write_file('WEB-INF', '')
    self.write_file('foo.jar', '')
    errors = []

    def ErrorFake(message):
        errors.append(message)
    with mock.patch.dict(ext_runtime._LOG_FUNCS, {'error': ErrorFake}):
        self.assertFalse(self.generate_configs())
    self.assertEqual(errors, ['Too many java artifacts to deploy (.jar, .war, or Java Web App).'])