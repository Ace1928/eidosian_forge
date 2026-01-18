import glob
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from os.path import join as pjoin
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
import pytest
from jupyter_core import paths
from jupyterlab import commands
from jupyterlab.commands import (
from jupyterlab.coreconfig import CoreConfig, _get_default_core_data
def test_install_extension(self):
    assert install_extension(self.mock_extension) is True
    path = pjoin(self.app_dir, 'extensions', '*.tgz')
    assert glob.glob(path)
    extensions = get_app_info()['extensions']
    name = self.pkg_names['extension']
    assert name in extensions
    assert check_extension(name)