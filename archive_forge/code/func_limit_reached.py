from time import time
from qiskit.utils import optionals as _optionals
def limit_reached(self):
    """Checks if a limit is reached."""
    if self.call_current is not None:
        self.call_current += 1
        if self.call_current > self.call_limit:
            return True
    if self.time_start is not None:
        self.time_current = time() - self.time_start
        if self.time_current > self.time_limit:
            return True
    return False