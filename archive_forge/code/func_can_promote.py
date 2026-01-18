from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple
from triad import SerializableRLock, to_uuid
from tune.concepts.flow import (
def can_promote(self, report: TrialReport) -> Tuple[bool, str]:
    reasons: List[str] = []
    if self._active:
        can_accept = self._parent.can_accept(report.trial)
        early_stop = self._parent._parent._trial_early_stop(report, self._history, self._parent._rungs)
        self._active = can_accept and (not early_stop)
        if not can_accept:
            reasons.append("can't accept new")
        if early_stop:
            reasons.append('trial early stop')
    if self._active:
        self._history.append(report)
        can_push = self._parent._rungs[report.rung].push(report)
        if not can_push:
            reasons.append('not best')
        return (can_push, ', '.join(reasons))
    return (False, ', '.join(reasons))