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
@pytest.mark.skipif(platform.system() == 'Windows', reason='running npm pack fails on windows CI')
def test_install_and_uninstall_pinned_folder(self):
    """
        Same as above test, but installs from a local folder instead of from npm.
        """
    base_dir = Path(self.tempdir())
    packages = [subprocess.run(['npm', 'pack', name], stdout=subprocess.PIPE, text=True, check=True, cwd=str(base_dir)).stdout.strip() for name in self.pinned_packages]
    shutil.unpack_archive(str(base_dir / packages[0]), str(base_dir / '1'))
    shutil.unpack_archive(str(base_dir / packages[1]), str(base_dir / '2'))
    self.pinned_packages = [str(base_dir / '1' / 'package'), str(base_dir / '2' / 'package')]
    self.test_install_and_uninstall_pinned()