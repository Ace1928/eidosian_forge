import os
import shutil
import sys
import tempfile
from subprocess import check_output
from flaky import flaky
import pytest
from traitlets.tests.utils import check_help_all_output
def test_help_output():
    """jupyter console --help-all works"""
    check_help_all_output('jupyter_console')