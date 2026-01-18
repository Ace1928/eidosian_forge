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
def test_app_dir(self):
    app_dir = self.tempdir()
    options = AppOptions(app_dir=app_dir)
    assert install_extension(self.mock_extension, app_options=options) is True
    path = pjoin(app_dir, 'extensions', '*.tgz')
    assert glob.glob(path)
    extensions = get_app_info(app_options=options)['extensions']
    ext_name = self.pkg_names['extension']
    assert ext_name in extensions
    assert check_extension(ext_name, app_options=options)
    assert uninstall_extension(self.pkg_names['extension'], app_options=options) is True
    path = pjoin(app_dir, 'extensions', '*.tgz')
    assert not glob.glob(path)
    extensions = get_app_info(app_options=options)['extensions']
    assert ext_name not in extensions
    assert not check_extension(ext_name, app_options=options)
    assert link_package(self.mock_package, app_options=options) is True
    linked = get_app_info(app_options=options)['linked_packages']
    pkg_name = self.pkg_names['package']
    assert pkg_name in linked
    assert check_extension(pkg_name, app_options=options)
    assert unlink_package(self.mock_package, app_options=options) is True
    linked = get_app_info(app_options=options)['linked_packages']
    assert pkg_name not in linked
    assert not check_extension(pkg_name, app_options=options)