from itertools import zip_longest
import json
import re
import os
import sys
from os.path import abspath, dirname, join
from filecmp import cmp
import subprocess
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import yaml_available
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tee import capture_output
from pyomo.common.tempfiles import TempfileManager
import pyomo.core
import pyomo.scripting.pyomo_main as main
from pyomo.opt import check_available_solvers
from io import StringIO
def test3_missing_model_object(self):
    self.pyomo('pmedian1.py pmedian.dat', root=join(currdir, 'test3'))
    self.compare_json(join(currdir, 'test3.jsn'), join(currdir, 'test1.txt'))