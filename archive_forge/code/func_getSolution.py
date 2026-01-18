from time import time
from qiskit.utils import optionals as _optionals
def getSolution(self, domains, constraints, vconstraints):
    """Wrap RecursiveBacktrackingSolver.getSolution to add the limits."""
    if self.call_limit is not None:
        self.call_current = 0
    if self.time_limit is not None:
        self.time_start = time()
    return super().getSolution(domains, constraints, vconstraints)