import os
import tempfile
from textwrap import dedent
import unittest
from unittest import mock
from numba.tests.support import (TestCase, temp_directory, override_env_config,
from numba.core import config
def inject_mock_cfg(self, location, cfg):
    """
        Injects a mock configuration at 'location'
        """
    tmpcfg = os.path.join(location, config._config_fname)
    with open(tmpcfg, 'wt') as f:
        yaml.dump(cfg, f, default_flow_style=False)