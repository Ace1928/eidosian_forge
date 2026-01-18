from itertools import zip_longest
import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.environ import (
def test_md14(self):
    try:
        md = DataPortal(1)
        self.fail('Expected RuntimeError')
    except RuntimeError:
        pass
    try:
        md = DataPortal(foo=True)
        self.fail('Expected ValueError')
    except ValueError:
        pass