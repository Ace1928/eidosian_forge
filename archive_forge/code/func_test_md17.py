from itertools import zip_longest
import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.environ import (
def test_md17(self):
    md = DataPortal()
    try:
        md[1, 2, 3, 4]
        self.fail('Expected IOError')
    except IOError:
        pass