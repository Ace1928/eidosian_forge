from os.path import join, dirname, abspath
import json
import pyomo.common.unittest as unittest
import pyomo.kernel as pmo
from pyomo.core.kernel.block import IBlock
from pyomo.core import Suffix, Var, Constraint, Objective
from pyomo.opt import ProblemFormat, SolverFactory, TerminationCondition
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
def generate_model(self, import_suffixes=[]):
    """Generate the model"""
    self._generate_model()
    self.test_suffixes = [] if self.disable_suffix_tests else import_suffixes
    if isinstance(self.model, IBlock):
        for suffix in self.test_suffixes:
            setattr(self.model, suffix, pmo.suffix(direction=pmo.suffix.IMPORT))
    else:
        for suffix in self.test_suffixes:
            setattr(self.model, suffix, Suffix(direction=Suffix.IMPORT))