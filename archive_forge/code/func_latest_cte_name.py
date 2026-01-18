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
@property
def latest_cte_name(self) -> str:
    if not self.expression.ctes:
        from_exp = self.expression.args['from']
        if from_exp.alias_or_name:
            return from_exp.alias_or_name
        table_alias = from_exp.find(exp.TableAlias)
        if not table_alias:
            raise RuntimeError(f'Could not find an alias name for this expression: {self.expression}')
        return table_alias.alias_or_name
    return self.expression.ctes[-1].alias