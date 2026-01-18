import itertools
import os
import uuid
from datetime import date
from pathlib import Path
from typing import Dict, Iterable
import submitit
from xformers.benchmarks.LRA.run_with_submitit import (
def grid_parameters(grid: Dict):
    """
    Yield all combinations of parameters in the grid (as a dict)
    """
    grid_copy = dict(grid)
    for k in grid_copy:
        if not isinstance(grid_copy[k], Iterable):
            grid_copy[k] = [grid_copy[k]]
    for p in itertools.product(*grid_copy.values()):
        yield dict(zip(grid.keys(), p))