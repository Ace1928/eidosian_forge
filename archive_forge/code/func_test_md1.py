from itertools import zip_longest
import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.environ import (
def test_md1(self):
    md = DataPortal()
    md.connect(filename=example_dir + 'A.tab')
    try:
        md.load()
        self.fail('Must specify a model')
    except ValueError:
        pass
    model = AbstractModel()
    try:
        md.load(model=model)
        self.fail('Expected ValueError')
    except ValueError:
        pass
    model.A = Set()