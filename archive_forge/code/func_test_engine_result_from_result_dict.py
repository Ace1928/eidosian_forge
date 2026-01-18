import datetime
from typing import Mapping
import pandas as pd
import cirq
import cirq_google as cg
import numpy as np
def test_engine_result_from_result_dict():
    res = cg.EngineResult(job_id='my_job_id', job_finished_time=_DT, params=None, measurements={'a': np.array([[0, 0], [1, 1]]), 'b': np.array([[0, 0, 0], [1, 1, 1]])})
    res2 = cirq.ResultDict(params=None, measurements={'a': np.array([[0, 0], [1, 1]]), 'b': np.array([[0, 0, 0], [1, 1, 1]])})
    assert res2 != res
    assert res != res2
    assert res == cg.EngineResult.from_result(res2, job_id='my_job_id', job_finished_time=_DT)