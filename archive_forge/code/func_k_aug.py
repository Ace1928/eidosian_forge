import os
from pyomo.environ import SolverFactory
from pyomo.common.tempfiles import TempfileManager
def k_aug(self, model, **kwargs):
    with InTempDir():
        results = self._k_aug.solve(model, **kwargs)
        for fname in known_files:
            if os.path.exists(fname):
                with open(fname, 'r') as fp:
                    self.data[fname] = fp.read()
    return results