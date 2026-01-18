import re
import os
from os.path import abspath, dirname, join
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.opt import check_available_solvers
from pyomo.scripting.pyomo_main import main
from pyomo.core import (
from pyomo.common.tee import capture_output
from io import StringIO
def onearg_rule(b, i):
    b.args = (i,)