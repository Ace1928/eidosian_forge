from itertools import zip_longest
import re
import sys
import os
from os.path import join
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import ProblemFormat, ConverterError, convert_problem
from pyomo.common import Executable
def test_error9(self):
    cmd = Executable('pico_convert').disable()
    try:
        ans = convert_problem((join(currdir, 'test4.nl'),), None, [ProblemFormat.cpxlp])
        self.fail("This test didn't fail, but pico_convert should not be defined.")
    except ConverterError:
        pass
    cmd = Executable('pico_convert').rehash()