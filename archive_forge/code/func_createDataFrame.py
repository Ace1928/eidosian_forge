from __future__ import annotations
import typing as t
import uuid
from collections import defaultdict
import sqlglot
from sqlglot import Dialect, expressions as exp
from sqlglot.dataframe.sql import functions as F
from sqlglot.dataframe.sql.dataframe import DataFrame
from sqlglot.dataframe.sql.readwriter import DataFrameReader
from sqlglot.dataframe.sql.types import StructType
from sqlglot.dataframe.sql.util import get_column_mapping_from_schema_input
from sqlglot.helper import classproperty
from sqlglot.optimizer import optimize
from sqlglot.optimizer.qualify_columns import quote_identifiers
def createDataFrame(self, data: t.Sequence[t.Union[t.Dict[str, ColumnLiterals], t.List[ColumnLiterals], t.Tuple]], schema: t.Optional[SchemaInput]=None, samplingRatio: t.Optional[float]=None, verifySchema: bool=False) -> DataFrame:
    from sqlglot.dataframe.sql.dataframe import DataFrame
    if samplingRatio is not None or verifySchema:
        raise NotImplementedError('Sampling Ratio and Verify Schema are not supported')
    if schema is not None and (not isinstance(schema, (StructType, str, list)) or (isinstance(schema, list) and (not isinstance(schema[0], str)))):
        raise NotImplementedError('Only schema of either list or string of list supported')
    if not data:
        raise ValueError('Must provide data to create into a DataFrame')
    column_mapping: t.Dict[str, t.Optional[str]]
    if schema is not None:
        column_mapping = get_column_mapping_from_schema_input(schema)
    elif isinstance(data[0], dict):
        column_mapping = {col_name.strip(): None for col_name in data[0]}
    else:
        column_mapping = {f'_{i}': None for i in range(1, len(data[0]) + 1)}
    data_expressions = [exp.tuple_(*map(lambda x: F.lit(x).expression, row if not isinstance(row, dict) else row.values())) for row in data]
    sel_columns = [F.col(name).cast(data_type).alias(name).expression if data_type is not None else F.col(name).expression for name, data_type in column_mapping.items()]
    select_kwargs = {'expressions': sel_columns, 'from': exp.From(this=exp.Values(expressions=data_expressions, alias=exp.TableAlias(this=exp.to_identifier(self._auto_incrementing_name), columns=[exp.to_identifier(col_name) for col_name in column_mapping])))}
    sel_expression = exp.Select(**select_kwargs)
    return DataFrame(self, sel_expression)