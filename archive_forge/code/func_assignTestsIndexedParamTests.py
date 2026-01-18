import math
import os
import sys
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.errors import PyomoException
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
from pyomo.core.base.param import _ParamData
from pyomo.core.base.set import _SetData
from pyomo.core.base.units_container import units, pint_available, UnitsError
from io import StringIO
def assignTestsIndexedParamTests(cls, problem_list):
    for val in problem_list:
        attrName = 'test_mutable_' + val[0] + '_expr'
        setattr(cls, attrName, createIndexedParamMethod(eval(val[0]), val[1], val[2]))