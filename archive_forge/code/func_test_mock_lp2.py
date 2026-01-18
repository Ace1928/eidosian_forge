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
def test_mock_lp2(self):
    arg = MockArg2()
    try:
        ans = convert_problem((arg,), None, [ProblemFormat.cpxlp])
    except ConverterError:
        err = sys.exc_info()[1]
        if not Executable('pico_convert'):
            return
        else:
            self.fail("Expected ApplicationError because pico_convert is not available: '%s'" % str(err))
    self.assertEqual(ans[0][0][-15:], 'pico_convert.lp')
    os.remove(ans[0][0])