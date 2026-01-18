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
def optimize_by_sha(objective: Any, dataset: TuneDataset, plan: List[Tuple[float, int]], checkpoint_path: str='', distributed: Optional[bool]=None, monitor: Any=None) -> StudyResult:
    _objective = parse_iterative_objective(objective)
    _monitor = parse_monitor(monitor)
    checkpoint_path = TUNE_OBJECT_FACTORY.get_path_or_temp(checkpoint_path)
    path = os.path.join(checkpoint_path, str(uuid4()))
    for budget, keep in plan:
        obj = _NonIterativeObjectiveWrapper(_objective, checkpoint_path=path, budget=budget)
        result = optimize_noniterative(obj, dataset, distributed=distributed, monitor=_monitor)
        dataset = result.next_tune_dataset(keep)
    return result