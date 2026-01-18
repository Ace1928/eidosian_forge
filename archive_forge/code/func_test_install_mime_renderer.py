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
def test_install_mime_renderer(self):
    install_extension(self.mock_mimeextension)
    name = self.pkg_names['mimeextension']
    assert name in get_app_info()['extensions']
    assert check_extension(name)
    assert uninstall_extension(name) is True
    assert name not in get_app_info()['extensions']
    assert not check_extension(name)