from datetime import datetime
from typing import Any, Callable, Dict, List, Set
from triad import SerializableRLock
from triad.utils.convert import to_timedelta
from tune.concepts.flow import (
def small_improvement(threshold: float, updates: int) -> SimpleNonIterativeStopper:
    assert updates > 0

    def func(current: TrialReport, updated: bool, reports: List[TrialReport]):
        if not updated:
            return False
        if len(reports) <= updates:
            return False
        diff = reports[-updates - 1].sort_metric - current.sort_metric
        return diff < threshold
    return SimpleNonIterativeStopper(func, log_best_only=True)