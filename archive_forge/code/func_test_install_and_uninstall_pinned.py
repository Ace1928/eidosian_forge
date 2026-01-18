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
def test_install_and_uninstall_pinned(self):
    """
        You should be able to install different versions of the same extension with different
        pinned names and uninstall them with those names.
        """
    NAMES = ['test-1', 'test-2']
    assert install_extension(self.pinned_packages[0], pin=NAMES[0])
    assert install_extension(self.pinned_packages[1], pin=NAMES[1])
    extensions = get_app_info()['extensions']
    assert NAMES[0] in extensions
    assert NAMES[1] in extensions
    assert check_extension(NAMES[0])
    assert check_extension(NAMES[1])
    assert uninstall_extension(NAMES[0])
    assert uninstall_extension(NAMES[1])
    extensions = get_app_info()['extensions']
    assert NAMES[0] not in extensions
    assert NAMES[1] not in extensions
    assert not check_extension(NAMES[0])
    assert not check_extension(NAMES[1])