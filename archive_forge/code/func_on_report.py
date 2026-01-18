import heapq
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Set
from triad import SerializableRLock
from triad.utils.convert import to_datetime
from tune._utils import to_base64
from tune.concepts.flow.trial import Trial
from tune.concepts.space.parameters import TuningParametersTemplate, to_template
from tune.constants import TUNE_REPORT, TUNE_REPORT_ID, TUNE_REPORT_METRIC
def on_report(self, report: TrialReport) -> bool:
    with self._lock:
        updated = False
        if self._best_report is None or report.sort_metric < self._best_report.sort_metric:
            self._best_report = report
            updated = True
        if updated or not self._new_best_only:
            self.log(report)
        return updated