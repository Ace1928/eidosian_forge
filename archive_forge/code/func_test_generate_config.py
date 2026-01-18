import os
import shutil
import sys
import tempfile
from subprocess import check_output
from flaky import flaky
import pytest
from traitlets.tests.utils import check_help_all_output
def test_generate_config():
    """jupyter console --generate-config works"""
    td = tempfile.mkdtemp()
    try:
        check_output([sys.executable, '-m', 'jupyter_console', '--generate-config'], env={'JUPYTER_CONFIG_DIR': td})
        assert os.path.isfile(os.path.join(td, 'jupyter_console_config.py'))
    finally:
        shutil.rmtree(td)