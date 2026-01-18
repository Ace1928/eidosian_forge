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
def optimize_by_hyperband(objective: Any, dataset: TuneDataset, plans: List[List[Tuple[float, int]]], checkpoint_path: str='', distributed: Optional[bool]=None, monitor: Any=None) -> StudyResult:
    _monitor = parse_monitor(monitor)
    weights = [float(p[0][1]) for p in plans]
    datasets = dataset.split(weights, seed=0)
    result: Any = None
    for d, plan in zip(datasets, plans):
        r = optimize_by_sha(objective=objective, dataset=d, plan=plan, checkpoint_path=checkpoint_path, distributed=distributed, monitor=_monitor)
        if result is None:
            result = r
        else:
            result.union_with(r)
    return result