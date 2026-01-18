import html
import json
from typing import Any, Dict, List, Optional
from IPython import get_ipython
from IPython.core.magic import Magics, cell_magic, magics_class, needs_local_scope
from IPython.display import HTML, display
from triad import ParamDict
from triad.utils.convert import to_instance
from triad.utils.pyarrow import _field_to_expression
from fugue import DataFrame, DataFrameDisplay, ExecutionEngine
from fugue import fsql as fugue_sql
from fugue import get_dataset_display, make_execution_engine
from fugue.dataframe import YieldedDataFrame
from fugue.exceptions import FugueSQLSyntaxError
@needs_local_scope
@cell_magic('fsql')
def fsql(self, line: str, cell: str, local_ns: Any=None) -> None:
    try:
        dag = fugue_sql('\n' + cell, local_ns, fsql_ignore_case=self._fsql_ignore_case)
    except FugueSQLSyntaxError as ex:
        raise FugueSQLSyntaxError(str(ex)).with_traceback(None) from None
    dag.run(self.get_engine(line, {} if local_ns is None else local_ns))
    for k, v in dag.yields.items():
        if isinstance(v, YieldedDataFrame):
            local_ns[k] = v.result
        else:
            local_ns[k] = v