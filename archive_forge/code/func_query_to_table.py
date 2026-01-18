import itertools
import logging
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Type
import ibis
from ibis import BaseBackend
from triad import assert_or_throw
from fugue import StructuredRawSQL
from fugue.bag import Bag, LocalBag
from fugue.collections.partition import (
from fugue.dataframe import DataFrame, DataFrames, LocalDataFrame
from fugue.dataframe.utils import get_join_schemas
from fugue.execution.execution_engine import ExecutionEngine, MapEngine, SQLEngine
from ._compat import IbisTable
from ._utils import to_ibis_schema
from .dataframe import IbisDataFrame
def query_to_table(self, statement: str, dfs: Dict[str, Any]) -> IbisTable:
    cte: List[str] = []
    for k, v in dfs.items():
        idf = self.to_df(v)
        cte.append(k + ' AS (' + idf.to_sql() + ')')
    if len(cte) > 0:
        sql = 'WITH ' + ',\n'.join(cte) + '\n' + statement
    else:
        sql = statement
    return self.backend.sql(sql)