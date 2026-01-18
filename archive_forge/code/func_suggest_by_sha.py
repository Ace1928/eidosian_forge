from typing import Any, List, Optional, Tuple
from fugue import FugueWorkflow
from fugue.exceptions import FugueDataFrameError
from triad import assert_or_throw
from tune._utils import from_base64
from tune.api.factory import TUNE_OBJECT_FACTORY, parse_logger
from tune.api.optimize import (
from tune.concepts.flow import TrialReport
from tune.concepts.logger import make_logger
from tune.concepts.space import Space
from tune.constants import TUNE_DATASET_DF_DEFAULT_NAME, TUNE_REPORT, TUNE_REPORT_METRIC
from tune.exceptions import TuneCompileError
def suggest_by_sha(objective: Any, space: Space, plan: List[Tuple[float, int]], train_df: Any=None, temp_path: str='', partition_keys: Optional[List[str]]=None, top_n: int=1, monitor: Any=None, distributed: Optional[bool]=None, execution_engine: Any=None, execution_engine_conf: Any=None) -> List[TrialReport]:
    assert_or_throw(not space.has_stochastic, TuneCompileError("space can't contain random parameters, use sample method before calling this function"))
    dag = FugueWorkflow()
    dataset = TUNE_OBJECT_FACTORY.make_dataset(dag, space, df=train_df, partition_keys=partition_keys, temp_path=temp_path)
    study = optimize_by_sha(objective=objective, dataset=dataset, plan=plan, checkpoint_path=temp_path, distributed=distributed, monitor=monitor)
    study.result(top_n).yield_dataframe_as('result')
    return _run(dag=dag, execution_engine=execution_engine, execution_engine_conf=execution_engine_conf)