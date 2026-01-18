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
def test_install_incompatible(self):
    with pytest.raises(ValueError) as excinfo:
        install_extension(self.mock_incompat)
    assert 'Conflicting Dependencies' in str(excinfo.value)
    assert not check_extension(self.pkg_names['incompat'])