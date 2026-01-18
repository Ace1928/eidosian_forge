from typing import Any, Callable, List, Optional, Union
from fugue import FugueWorkflow
from triad import assert_or_throw, conditional_dispatcher
from tune.concepts.dataset import TuneDataset, TuneDatasetBuilder
from tune.concepts.flow import Monitor
from tune.concepts.space import Space
from tune.concepts.logger import MetricLogger
from tune.constants import (
from tune.exceptions import TuneCompileError
from tune.iterative.objective import IterativeObjectiveFunc
from tune.noniterative.convert import to_noniterative_objective
from tune.noniterative.objective import (
from tune.noniterative.stopper import NonIterativeStopper
def set_temp_path(self, path: str) -> None:
    self._tmp = path