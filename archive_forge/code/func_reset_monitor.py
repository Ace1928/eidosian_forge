from typing import Any, Callable, Dict, Optional
from tune.concepts.flow.report import TrialReport
from tune.concepts.flow.trial import Trial
def reset_monitor(self, monitor: Optional['Monitor']=None) -> None:
    self._trial_judge_monitor = monitor or Monitor()