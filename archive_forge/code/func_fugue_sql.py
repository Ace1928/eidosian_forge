from typing import Any, Dict, Tuple, Optional
from triad.utils.convert import get_caller_global_local_vars
from fugue.dataframe import AnyDataFrame
from fugue.exceptions import FugueSQLError
from fugue.execution import AnyExecutionEngine
from fugue.execution.api import get_current_conf
from ..constants import (
from .workflow import FugueSQLWorkflow
def fugue_sql(query: str, *args: Any, fsql_ignore_case: Optional[bool]=None, fsql_dialect: Optional[str]=None, engine: AnyExecutionEngine=None, engine_conf: Any=None, as_fugue: bool=False, as_local: bool=False, **kwargs: Any) -> AnyDataFrame:
    """Simplified Fugue SQL interface. This function can still take multiple dataframe
    inputs but will always return the last generated dataframe in the SQL workflow. And
    ``YIELD`` should NOT be used with this function. If you want to use Fugue SQL to
    represent the full workflow, or want to see more Fugue SQL examples,
    please read :func:`~.fugue_sql_flow`.

    :param query: the Fugue SQL string (can be a jinja template)
    :param args: variables related to the SQL string
    :param fsql_ignore_case: whether to ignore case when parsing the SQL string,
        defaults to None (it depends on the engine/global config).
    :param fsql_dialect: the dialect of this fsql,
        defaults to None (it depends on the engine/global config).
    :param kwargs: variables related to the SQL string
    :param engine: an engine like object, defaults to None
    :param engine_conf: the configs for the engine, defaults to None
    :param as_fugue: whether to force return a Fugue DataFrame, defaults to False
    :param as_local: whether return a local dataframe, defaults to False

    :return: the result dataframe

    .. note::

        This function is different from :func:`~fugue.api.raw_sql` which directly
        sends the query to the execution engine to run. This function parses the query
        based on Fugue SQL syntax, creates a
        :class:`~fugue.sql.workflow.FugueSQLWorkflow` which
        could contain multiple raw SQLs plus other operations, and runs and returns
        the last dataframe generated in the workflow.

        This function allows you to parameterize the SQL in a more elegant way. The
        data tables referred in the query can either be automatically extracted from the
        local variables or be specified in the arguments.

    .. caution::

        Currently, we have not unified the dialects of different SQL backends. So there
        can be some slight syntax differences when you switch between backends.
        In addition, we have not unified the UDFs cross different backends, so you
        should be careful to use uncommon UDFs belonging to a certain backend.

        That being said, if you keep your SQL part general and leverage Fugue extensions
        (transformer, creator, processor, outputter, etc.) appropriately, it should be
        easy to write backend agnostic Fugue SQL.

        We are working on unifying the dialects of different SQLs, it should be
        available in the future releases. Regarding unifying UDFs, the effort is still
        unclear.

    .. code-block:: python

        import pandas as pd
        import fugue.api as fa

        def tr(df:pd.DataFrame) -> pd.DataFrame:
            return df.assign(c=2)

        input = pd.DataFrame([[0,1],[3.4]], columns=["a","b"])

        with fa.engine_context("duckdb"):
            res = fa.fugue_sql('''
            SELECT * FROM input WHERE a<{{x}}
            TRANSFORM USING tr SCHEMA *,c:int
            ''', x=2)
            assert fa.as_array(res) == [[0,1,2]]
    """
    dag = _build_dag(query, fsql_ignore_case=fsql_ignore_case, fsql_dialect=fsql_dialect, args=args, kwargs=kwargs)
    if dag.last_df is not None:
        dag.last_df.yield_dataframe_as('result', as_local=as_local)
    else:
        raise FugueSQLError(f'no dataframe to output from\n{query}')
    res = dag.run(engine, engine_conf)
    return res['result'] if as_fugue else res['result'].native_as_df()