from typing import Any, Callable, Dict, Optional
from tune.concepts.flow.report import TrialReport
from tune.concepts.flow.trial import Trial
def on_judge(self, decision: TrialDecision) -> None:
    pass