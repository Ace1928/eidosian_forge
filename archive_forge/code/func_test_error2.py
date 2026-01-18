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
def test_error2(self):
    try:
        convert_problem(('test4.nl', 'tmp.nl'), ProblemFormat.nl, [ProblemFormat.mps])
        self.fail('Expected ConverterError exception')
    except ConverterError:
        pass