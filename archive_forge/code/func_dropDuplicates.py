from __future__ import annotations
import functools
import logging
import typing as t
import zlib
from copy import copy
import sqlglot
from sqlglot import Dialect, expressions as exp
from sqlglot.dataframe.sql import functions as F
from sqlglot.dataframe.sql.column import Column
from sqlglot.dataframe.sql.group import GroupedData
from sqlglot.dataframe.sql.normalize import normalize
from sqlglot.dataframe.sql.operations import Operation, operation
from sqlglot.dataframe.sql.readwriter import DataFrameWriter
from sqlglot.dataframe.sql.transforms import replace_id_value
from sqlglot.dataframe.sql.util import get_tables_from_expression_with_join
from sqlglot.dataframe.sql.window import Window
from sqlglot.helper import ensure_list, object_to_dict, seq_get
@operation(Operation.SELECT)
def dropDuplicates(self, subset: t.Optional[t.List[str]]=None):
    if not subset:
        return self.distinct()
    column_names = ensure_list(subset)
    window = Window.partitionBy(*column_names).orderBy(*column_names)
    return self.copy().withColumn('row_num', F.row_number().over(window)).where(F.col('row_num') == F.lit(1)).drop('row_num')