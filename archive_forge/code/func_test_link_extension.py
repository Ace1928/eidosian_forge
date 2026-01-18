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
def test_link_extension(self):
    path = self.mock_extension
    name = self.pkg_names['extension']
    link_package(path)
    linked = get_app_info()['linked_packages']
    assert name not in linked
    assert name in get_app_info()['extensions']
    assert check_extension(name)
    assert unlink_package(path) is True
    linked = get_app_info()['linked_packages']
    assert name not in linked
    assert name not in get_app_info()['extensions']
    assert not check_extension(name)