from typing import Any, Dict, List, Optional
from triad.utils.assertion import assert_or_throw
from ..collections.yielded import Yielded
from ..constants import FUGUE_CONF_WORKFLOW_EXCEPTION_INJECT
from ..dataframe import DataFrame, AnyDataFrame
from ..dataframe.api import get_native_as_df
from ..exceptions import FugueInterfacelessError, FugueWorkflowCompileError
from ..execution import make_execution_engine
from .workflow import FugueWorkflow
def raw_sql(*statements: Any, engine: Any=None, engine_conf: Any=None, as_fugue: bool=False, as_local: bool=False) -> AnyDataFrame:
    """Run raw SQL on the execution engine

    :param statements: a sequence of sub-statements in string
        or dataframe-like objects
    :param engine: an engine like object, defaults to None
    :param engine_conf: the configs for the engine, defaults to None
    :param as_fugue: whether to force return a Fugue DataFrame, defaults to False
    :param as_local: whether return a local dataframe, defaults to False

    :return: the result dataframe

    .. caution::

        Currently, only ``SELECT`` statements are supported

    .. admonition:: Examples

        .. code-block:: python

            import pandas as pd
            import fugue.api as fa

            with fa.engine_context("duckdb"):
                a = fa.as_fugue_df([[0,1]], schema="a:long,b:long")
                b = pd.DataFrame([[0,10]], columns=["a","b"])
                c = fa.raw_sql("SELECT * FROM",a,"UNION SELECT * FROM",b)
                fa.as_pandas(c)
    """
    dag = FugueWorkflow(compile_conf={FUGUE_CONF_WORKFLOW_EXCEPTION_INJECT: 0})
    sp: List[Any] = []
    infer_by: List[Any] = []
    inputs: Dict[int, Any] = {}
    for x in statements:
        if isinstance(x, str):
            sp.append(x)
        elif id(x) in inputs:
            sp.append(inputs[id(x)])
        else:
            inputs[id(x)] = dag.create(x)
            sp.append(inputs[id(x)])
            infer_by.append(x)
    engine = make_execution_engine(engine, engine_conf, infer_by=infer_by)
    dag.select(*sp).yield_dataframe_as('result', as_local=as_local)
    res = dag.run(engine)
    return res['result'] if as_fugue else get_native_as_df(res['result'])