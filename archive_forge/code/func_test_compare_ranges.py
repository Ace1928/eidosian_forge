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
def test_compare_ranges(self):
    assert _compare_ranges('^1 || ^2', '^1') == 0
    assert _compare_ranges('^1 || ^2', '^2 || ^3') == 0
    assert _compare_ranges('^1 || ^2', '^3 || ^4') == 1
    assert _compare_ranges('^3 || ^4', '^1 || ^2') == -1
    assert _compare_ranges('^2 || ^3', '^1 || ^4') is None