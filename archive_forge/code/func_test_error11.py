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
def test_error11(self):
    try:
        ans = convert_problem((join(currdir, 'test3.mod'), join(currdir, 'test5.dat')), None, [ProblemFormat.cpxlp])
        self.fail("Expected ConverterError exception because we provided a MOD file with a 'data;' declaration")
    except ApplicationError:
        err = sys.exc_info()[1]
        if Executable('glpsol'):
            self.fail("Expected ApplicationError because glpsol is not available: '%s'" % str(err))
        return
    except ConverterError:
        pass