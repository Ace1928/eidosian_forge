import os
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import yaml, yaml_available, yaml_load_args
from pyomo.common.fileutils import this_file_dir, PYOMO_ROOT_DIR
import pyomo.opt
import pyomo.scripting.pyomo_main as pyomo_main
from pyomo.scripting.util import cleanup
import pyomo.environ
def test_munson1a(self):
    self.problem = 'test_munson1a'
    self.run_solver(os.path.join(exdir, 'munson1a.py'))
    self.check('munson1a', self.solver)