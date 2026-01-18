import os
import tempfile
from textwrap import dedent
import unittest
from unittest import mock
from numba.tests.support import (TestCase, temp_directory, override_env_config,
from numba.core import config
def mock_cfg_location(self):
    """
        Creates a mock launch location.
        Returns the location path.
        """
    return tempfile.mkdtemp(dir=self.tmppath)