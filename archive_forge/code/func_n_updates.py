from datetime import datetime
from typing import Any, Callable, Dict, List, Set
from triad import SerializableRLock
from triad.utils.convert import to_timedelta
from tune.concepts.flow import (
def n_updates(n: int) -> SimpleNonIterativeStopper:

    def func(current: TrialReport, updated: bool, reports: List[TrialReport]):
        return len(reports) >= n
    return SimpleNonIterativeStopper(func, log_best_only=True)