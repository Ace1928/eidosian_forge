from typing import Any, Dict, Optional
from uuid import uuid4
from tune.concepts.flow.report import TrialReport
def log_report(self, report: TrialReport, log_params: bool=False, log_metadata: bool=False, extract_metrics: bool=False) -> None:
    """Log information from a TrialReport

        :param report: the report
        :param log_params: whether to log the hyperparameter from the report,
            defaults to True
        :param log_metadata: whether to log the metadata from the report,
            defaults to True
        :param extract_metrics: whether to extract more metrics from the report
            and log as metric_, defaults to True
        """
    all_metrics = {'OBJECTIVE_METRIC': report.metric}
    if extract_metrics:
        all_metrics.update({k: float(v) for k, v in report.metadata.items() if isinstance(v, (int, float))})
    self.log_metrics(all_metrics)
    if log_params:
        self.log_params(report.trial.params.simple_value)
    if log_metadata:
        self.log_metadata(report.metadata)