import functools
import importlib
import os
import platform
import subprocess
import sys
import pytest
from matplotlib import _c_internal_utils
from matplotlib.testing import subprocess_run_helper
def legitimate_quit():
    root.quit()
    success.append(True)