from itertools import zip_longest
import json
import re
import glob
import subprocess
import os
from os.path import join
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import attempt_import
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager
import pyomo.scripting.pyomo_main as main
def pyomo(self, cmd):
    os.chdir(currdir)
    output = main.main(['convert', '--logging=quiet', '-c'] + cmd)
    return output