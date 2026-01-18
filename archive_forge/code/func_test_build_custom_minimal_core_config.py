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
def test_build_custom_minimal_core_config(self):
    default_config = CoreConfig()
    core_config = CoreConfig()
    core_config.clear_packages()
    logger = logging.getLogger('jupyterlab_test_logger')
    logger.setLevel('DEBUG')
    app_dir = self.tempdir()
    options = AppOptions(app_dir=app_dir, core_config=core_config, logger=logger, use_sys_dir=False)
    extensions = ('@jupyterlab/application-extension', '@jupyterlab/apputils-extension')
    singletons = ('@jupyterlab/application', '@jupyterlab/apputils', '@jupyterlab/coreutils', '@jupyterlab/services')
    for name in extensions:
        semver = default_config.extensions[name]
        core_config.add(name, semver, extension=True)
    for name in singletons:
        semver = default_config.singletons[name]
        core_config.add(name, semver)
    assert install_extension(self.mock_extension, app_options=options) is True
    build(app_options=options)
    entry = pjoin(app_dir, 'static', 'index.out.js')
    with open(entry) as fid:
        data = fid.read()
    assert self.pkg_names['extension'] in data
    pkg = pjoin(app_dir, 'static', 'package.json')
    with open(pkg) as fid:
        data = json.load(fid)
    assert sorted(data['jupyterlab']['extensions'].keys()) == ['@jupyterlab/application-extension', '@jupyterlab/apputils-extension', '@jupyterlab/mock-extension']
    assert data['jupyterlab']['mimeExtensions'] == {}
    for pkg in data['jupyterlab']['singletonPackages']:
        if pkg.startswith('@jupyterlab/'):
            assert pkg in singletons