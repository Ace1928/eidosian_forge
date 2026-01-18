import contextlib
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
from unittest import skip
from ctypes import *
import numpy as np
import llvmlite.binding as ll
from numba.core import utils
from numba.tests.support import (TestCase, tag, import_dynamic, temp_directory,
import unittest
@contextlib.contextmanager
def check_cc_compiled(self, cc):
    cc.output_dir = self.tmpdir
    cc.compile()
    with self.check_c_ext(self.tmpdir, cc.name) as lib:
        yield lib