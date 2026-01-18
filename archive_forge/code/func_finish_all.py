from multiprocessing import Pool
import time
from ase.io import write, read
def finish_all(self):
    """Checks that all calculations are finished, if not
        wait and check again. Return when all are finished."""
    while len(self.results) > 0:
        self._cleanup()
        time.sleep(2.0)