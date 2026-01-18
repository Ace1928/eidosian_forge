from typing import Any, Callable, Dict, Optional
from tune.concepts.flow.report import TrialReport
from tune.concepts.flow.trial import Trial
def judge(self, report: TrialReport) -> TrialDecision:
    self.monitor.on_report(report)
    return TrialDecision(report, 0.0, False)