from typing import List
import pandas as pd
from fugue import FugueWorkflow
from pytest import raises
from tune import optimize_noniterative, suggest_for_noniterative_objective
from tune.concepts.dataset import TuneDatasetBuilder
from tune.concepts.flow import Monitor
from tune.concepts.space import Grid, Space
from tune.constants import TUNE_REPORT, TUNE_REPORT_METRIC
from tune.exceptions import TuneInterrupted
from tune.noniterative.convert import to_noniterative_objective
from tune.noniterative.stopper import n_samples
def objective4(a: float, b: pd.DataFrame) -> float:
    if a == -2:
        raise ValueError('expected')
    return a ** 2 + b.shape[0]