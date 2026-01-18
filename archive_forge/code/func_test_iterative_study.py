from typing import Any, Dict, Iterable
import numpy as np
from fugue.workflow.workflow import FugueWorkflow
from tune import Space, Trial, TrialDecision, TrialReport
from tune.constants import TUNE_REPORT_METRIC
from tune.concepts.dataset import TuneDatasetBuilder
from tune.iterative.objective import IterativeObjectiveFunc
from tune.iterative.study import IterativeStudy
from tune.concepts.flow import TrialJudge
def test_iterative_study(tmpdir):

    def assert_metric(df: Iterable[Dict[str, Any]], metric: float) -> None:
        for row in df:
            assert row[TUNE_REPORT_METRIC] < metric
    study = IterativeStudy(F(), str(tmpdir))
    space = sum((Space(a=a, b=b) for a, b in [(1.1, 0.2), (0.8, -0.2), (1.2, -0.1), (0.7, 0.3), (1.0, 1.5)]))
    dag = FugueWorkflow()
    dataset = TuneDatasetBuilder(space, str(tmpdir)).build(dag)
    result = study.optimize(dataset, J([1, 2, 3, 4]))
    result.result(1).show()
    result.result(1).output(assert_metric, params=dict(metric=-2.8))
    dag.run()