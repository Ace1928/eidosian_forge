import os
import tempfile
from textwrap import dedent
import unittest
from unittest import mock
from numba.tests.support import (TestCase, temp_directory, override_env_config,
from numba.core import config
def create_config_effect(self, cfg):
    """
        Returns a config "original" from a location with no config file
        and then the impact of applying the supplied cfg dictionary as
        a config file at a location in the returned "current".
        """
    original_cwd = os.getcwd()
    launch_dir = self.mock_cfg_location()
    os.chdir(launch_dir)
    with override_env_config('_', '_'):
        original = self.get_settings()
    self.inject_mock_cfg(launch_dir, cfg)
    try:
        with override_env_config('_', '_'):
            current = self.get_settings()
    finally:
        os.chdir(original_cwd)
    return (original, current)