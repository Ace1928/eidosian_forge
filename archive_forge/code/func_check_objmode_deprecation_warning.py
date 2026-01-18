import os
import subprocess
import sys
import warnings
import numpy as np
import unittest
from numba import jit
from numba.core.errors import (
from numba.core import errors
from numba.tests.support import ignore_internal_warnings
def check_objmode_deprecation_warning(self, w):
    msg = 'Fall-back from the nopython compilation path to the object mode compilation path has been detected'
    self.assertEqual(w.category, NumbaDeprecationWarning)
    self.assertIn(msg, str(w.message))