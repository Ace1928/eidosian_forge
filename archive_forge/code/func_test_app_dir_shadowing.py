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
def test_app_dir_shadowing(self):
    app_dir = self.tempdir()
    sys_dir = self.app_dir
    app_options = AppOptions(app_dir=app_dir)
    if os.path.exists(sys_dir):
        os.removedirs(sys_dir)
    assert install_extension(self.mock_extension) is True
    sys_path = pjoin(sys_dir, 'extensions', '*.tgz')
    assert glob.glob(sys_path)
    app_path = pjoin(app_dir, 'extensions', '*.tgz')
    assert not glob.glob(app_path)
    extensions = get_app_info(app_options=app_options)['extensions']
    ext_name = self.pkg_names['extension']
    assert ext_name in extensions
    assert check_extension(ext_name, app_options=app_options)
    assert install_extension(self.mock_extension, app_options=app_options) is True
    assert glob.glob(app_path)
    extensions = get_app_info(app_options=app_options)['extensions']
    assert ext_name in extensions
    assert check_extension(ext_name, app_options=app_options)
    assert uninstall_extension(self.pkg_names['extension'], app_options=app_options) is True
    assert not glob.glob(app_path)
    assert glob.glob(sys_path)
    extensions = get_app_info(app_options=app_options)['extensions']
    assert ext_name in extensions
    assert check_extension(ext_name, app_options=app_options)
    assert uninstall_extension(self.pkg_names['extension'], app_options=app_options) is True
    assert not glob.glob(app_path)
    assert not glob.glob(sys_path)
    extensions = get_app_info(app_options=app_options)['extensions']
    assert ext_name not in extensions
    assert not check_extension(ext_name, app_options=app_options)