from typing import Any, Callable, Dict
import ibis
from fugue import DataFrame, DataFrames, Processor, WorkflowDataFrame
from fugue.exceptions import FugueWorkflowCompileError
from fugue.workflow.workflow import WorkflowDataFrames
from triad import assert_or_throw, extension_method
from ._utils import LazyIbisObject, _materialize
from .execution.ibis_engine import parse_ibis_engine
from ._compat import IbisTable
@extension_method
def as_ibis(df: WorkflowDataFrame) -> IbisTable:
    """Convert the Fugue workflow dataframe to an ibis table for ibis
    operations.

    :param df: the Fugue workflow dataframe
    :return: the object representing the ibis table

    .. admonition:: Examples

        .. code-block:: python

            # non-magical approach
            import fugue as FugueWorkflow
            from fugue_ibis import as_ibis, as_fugue

            dag = FugueWorkflow()
            df1 = dag.df([[0]], "a:int")
            df2 = dag.df([[1]], "a:int")
            idf1 = as_ibis(df1)
            idf2 = as_ibis(df2)
            idf3 = idf1.union(idf2)
            result = idf3.mutate(b=idf3.a+1)
            as_fugue(result).show()

        .. code-block:: python

            # magical approach
            import fugue as FugueWorkflow
            import fugue_ibis  # must import

            dag = FugueWorkflow()
            idf1 = dag.df([[0]], "a:int").as_ibis()
            idf2 = dag.df([[1]], "a:int").as_ibis()
            idf3 = idf1.union(idf2)
            result = idf3.mutate(b=idf3.a+1).as_fugue()
            result.show()

    .. note::

        The magic is that when importing ``fugue_ibis``, the functions
        ``as_ibis`` and ``as_fugue`` are added to the correspondent classes
        so you can use them as if they are parts of the original classes.

        This is an idea similar to patching. Ibis uses this programming model
        a lot. Fugue provides this as an option.

    .. note::

        The returned object is not really a ``TableExpr``, it's a 'super lazy'
        object that will be translated into ``TableExpr`` at run time.
        This is because to compile an ibis execution graph, the input schemas
        must be known. However, in Fugue, this is not always true. For example
        if the previous step is to pivot a table, then the output schema can be
        known at runtime. So in order to be a part of Fugue, we need to be able to
        construct ibis expressions before knowing the input schemas.
    """
    return LazyIbisObject(df)