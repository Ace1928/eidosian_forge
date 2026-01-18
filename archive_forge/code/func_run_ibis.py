from typing import Any, Callable, Dict
import ibis
from fugue import DataFrame, DataFrames, Processor, WorkflowDataFrame
from fugue.exceptions import FugueWorkflowCompileError
from fugue.workflow.workflow import WorkflowDataFrames
from triad import assert_or_throw, extension_method
from ._utils import LazyIbisObject, _materialize
from .execution.ibis_engine import parse_ibis_engine
from ._compat import IbisTable
def run_ibis(ibis_func: Callable[[ibis.BaseBackend], IbisTable], ibis_engine: Any=None, **dfs: WorkflowDataFrame) -> WorkflowDataFrame:
    """Run an ibis workflow wrapped in ``ibis_func``

    :param ibis_func: the function taking in an ibis backend, and returning
      an Ibis TableExpr
    :param ibis_engine: an object that together with |ExecutionEngine|
      can determine :class:`~fugue_ibis.execution.ibis_engine.IbisEngine`
      , defaults to None
    :param dfs: dataframes in the same workflow
    :return: the output workflow dataframe

    .. admonition:: Examples

        .. code-block:: python

            import fugue as FugueWorkflow
            from fugue_ibis import run_ibis

            def func(backend):
                t = backend.table("tb")
                return t.mutate(b=t.a+1)

            dag = FugueWorkflow()
            df = dag.df([[0]], "a:int")
            result = run_ibis(func, tb=df)
            result.show()
    """
    wdfs = WorkflowDataFrames(**dfs)
    return wdfs.workflow.process(wdfs, using=_IbisProcessor, params=dict(ibis_func=ibis_func, ibis_engine=ibis_engine))