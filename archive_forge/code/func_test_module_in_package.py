from test import support
from test.support import import_helper
import_helper.import_module('_multiprocessing')
import importlib
import importlib.machinery
import unittest
import sys
import os
import os.path
import py_compile
from test.support import os_helper
from test.support.script_helper import (
import multiprocess as multiprocessing
import_helper.import_module('multiprocess.synchronize')
import sys
import time
from multiprocess import Pool, set_start_method
import sys
import time
from multiprocess import Pool, set_start_method
import sys, os.path, runpy
def test_module_in_package(self):
    with os_helper.temp_dir() as script_dir:
        pkg_dir = os.path.join(script_dir, 'test_pkg')
        make_pkg(pkg_dir)
        script_name = _make_test_script(pkg_dir, 'check_sibling')
        launch_name = _make_launch_script(script_dir, 'launch', 'test_pkg.check_sibling')
        self._check_script(launch_name)