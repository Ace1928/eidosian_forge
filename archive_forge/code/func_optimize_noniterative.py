import os
from typing import Any, Callable, List, Optional, Tuple
from uuid import uuid4
from triad import FileSystem
from tune.api.factory import (
from tune.concepts.dataset import StudyResult, TuneDataset
from tune.concepts.flow import TrialReport
from tune.iterative.asha import ASHAJudge, RungHeap
from tune.iterative.sha import _NonIterativeObjectiveWrapper
from tune.iterative.study import IterativeStudy
from tune.noniterative.study import NonIterativeStudy
def optimize_noniterative(objective: Any, dataset: TuneDataset, optimizer: Any=None, distributed: Optional[bool]=None, logger: Any=None, monitor: Any=None, stopper: Any=None, stop_check_interval: Any=None) -> StudyResult:
    _objective = parse_noniterative_objective(objective)
    _optimizer = parse_noniterative_local_optimizer(optimizer)
    _stopper = parse_noniterative_stopper(stopper)
    _monitor = parse_monitor(monitor)
    study = NonIterativeStudy(_objective, _optimizer)
    return study.optimize(dataset, distributed=distributed, monitor=_monitor, stopper=_stopper, stop_check_interval=stop_check_interval, logger=logger)