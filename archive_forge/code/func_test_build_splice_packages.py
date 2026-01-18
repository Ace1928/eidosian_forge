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
@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(DEV_DIR), reason='Not in git checkout')
def test_build_splice_packages(self):
    app_options = AppOptions(splice_source=True)
    assert install_extension(self.mock_extension) is True
    build(app_options=app_options)
    assert '-spliced' in get_app_version(app_options)
    entry = pjoin(self.app_dir, 'staging', 'build', 'index.out.js')
    with open(entry) as fid:
        data = fid.read()
    assert self.pkg_names['extension'] in data
    entry = pjoin(self.app_dir, 'static', 'index.out.js')
    with open(entry) as fid:
        data = fid.read()
    assert self.pkg_names['extension'] in data