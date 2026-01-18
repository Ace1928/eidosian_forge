from typing import Any, Dict, Iterable
from fugue import FugueWorkflow
from tune.concepts.dataset import TuneDatasetBuilder
from tune.iterative.objective import IterativeObjectiveFunc
from tune import optimize_by_sha, optimize_by_hyperband
from tune.concepts.space import Grid, Space
from tune.concepts.flow import TrialReport
from tune.constants import TUNE_REPORT_METRIC
def test_hyperband(tmpdir):

    def assert_metric(df: Iterable[Dict[str, Any]], metric: float, ct: int) -> None:
        n = 0
        for row in df:
            if metric > 0:
                assert row[TUNE_REPORT_METRIC] == metric
            n += 1
        assert n == ct
    space = Space(a=Grid(0, 1, 2, 3))
    dag = FugueWorkflow()
    dataset = TuneDatasetBuilder(space, str(tmpdir)).build(dag)
    obj = F()
    res = optimize_by_hyperband(obj, dataset, plans=[[[1.0, 3], [1.0, 2], [1.0, 1], [1.0, 1]], [[2.0, 2], [1.0, 1], [1.0, 1]]], checkpoint_path=str(tmpdir))
    res.result().output(assert_metric, dict(metric=0.0, ct=2))
    res.result(1).output(assert_metric, dict(metric=1.0, ct=1))
    dag.run()