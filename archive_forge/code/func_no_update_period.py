from datetime import datetime
from typing import Any, Callable, Dict, List, Set
from triad import SerializableRLock
from triad.utils.convert import to_timedelta
from tune.concepts.flow import (
def no_update_period(period: Any) -> SimpleNonIterativeStopper:
    _interval = to_timedelta(period)

    def func(current: TrialReport, updated: bool, reports: List[TrialReport]):
        if updated or len(reports) == 0:
            return False
        return datetime.now() - reports[-1].log_time > _interval
    return SimpleNonIterativeStopper(func, log_best_only=True)