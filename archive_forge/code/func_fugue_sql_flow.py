from typing import Any, Dict, Tuple, Optional
from triad.utils.convert import get_caller_global_local_vars
from fugue.dataframe import AnyDataFrame
from fugue.exceptions import FugueSQLError
from fugue.execution import AnyExecutionEngine
from fugue.execution.api import get_current_conf
from ..constants import (
from .workflow import FugueSQLWorkflow
def fugue_sql_flow(query: str, *args: Any, fsql_ignore_case: Optional[bool]=None, fsql_dialect: Optional[str]=None, **kwargs: Any) -> FugueSQLWorkflow:
    """Fugue SQL full functional interface. This function allows full workflow
    definition using Fugue SQL, and it allows multiple outputs using ``YIELD``.

    :param query: the Fugue SQL string (can be a jinja template)
    :param args: variables related to the SQL string
    :param fsql_ignore_case: whether to ignore case when parsing the SQL string,
        defaults to None (it depends on the engine/global config).
    :param fsql_dialect: the dialect of this fsql,
        defaults to None (it depends on the engine/global config).
    :param kwargs: variables related to the SQL string
    :return: the translated Fugue workflow

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

        import fugue.api.fugue_sql_flow as fsql
        import fugue.api as fa

        # Basic case
        fsql('''
        CREATE [[0]] SCHEMA a:int
        PRINT
        ''').run()

        # With external data sources
        df = pd.DataFrame([[0],[1]], columns=["a"])
        fsql('''
        SELECT * FROM df WHERE a=0
        PRINT
        ''').run()

        # With external variables
        df = pd.DataFrame([[0],[1]], columns=["a"])
        t = 1
        fsql('''
        SELECT * FROM df WHERE a={{t}}
        PRINT
        ''').run()

        # The following is the explicit way to specify variables and datafrems
        # (recommended)
        df = pd.DataFrame([[0],[1]], columns=["a"])
        t = 1
        fsql('''
        SELECT * FROM df WHERE a={{t}}
        PRINT
        ''', df=df, t=t).run()

        # Using extensions
        def dummy(df:pd.DataFrame) -> pd.DataFrame:
            return df

        fsql('''
        CREATE [[0]] SCHEMA a:int
        TRANSFORM USING dummy SCHEMA *
        PRINT
        ''').run()

        # It's recommended to provide full path of the extension inside
        # Fugue SQL, so the SQL definition and exeuction can be more
        # independent from the extension definition.

        # Run with different execution engines
        sql = '''
        CREATE [[0]] SCHEMA a:int
        TRANSFORM USING dummy SCHEMA *
        PRINT
        '''

        fsql(sql).run(spark_session)
        fsql(sql).run("dask")

        with fa.engine_context("duckdb"):
            fsql(sql).run()

        # Passing dataframes between fsql calls
        result = fsql('''
        CREATE [[0]] SCHEMA a:int
        YIELD DATAFRAME AS x

        CREATE [[1]] SCHEMA a:int
        YIELD DATAFRAME AS y
        ''').run(DaskExecutionEngine)

        fsql('''
        SELECT * FROM x
        UNION
        SELECT * FROM y
        UNION
        SELECT * FROM z

        PRINT
        ''', result, z=pd.DataFrame([[2]], columns=["z"])).run()

        # Get framework native dataframes
        result["x"].native  # Dask dataframe
        result["y"].native  # Dask dataframe
        result["x"].as_pandas()  # Pandas dataframe

        # Use lower case fugue sql
        df = pd.DataFrame([[0],[1]], columns=["a"])
        t = 1
        fsql('''
        select * from df where a={{t}}
        print
        ''', df=df, t=t, fsql_ignore_case=True).run()
    """
    dag = _build_dag(query, fsql_ignore_case=fsql_ignore_case, fsql_dialect=fsql_dialect, args=args, kwargs=kwargs)
    return dag