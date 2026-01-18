import os
from pyomo.environ import SolverFactory
from pyomo.common.tempfiles import TempfileManager
def dot_sens(self, model, **kwargs):
    with InTempDir():
        for fname, contents in self.data.items():
            if contents is not None:
                with open(fname, 'w') as fp:
                    fp.write(contents)
        results = self._dot_sens.solve(model, **kwargs)
        for fname in known_files:
            if os.path.exists(fname):
                with open(fname, 'r') as fp:
                    self.data[fname] = fp.read()
    return results