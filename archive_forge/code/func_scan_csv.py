import ast
import collections
import itertools
import math
from sqlglot import exp, generator, planner, tokens
from sqlglot.dialects.dialect import Dialect, inline_array_sql
from sqlglot.errors import ExecuteError
from sqlglot.executor.context import Context
from sqlglot.executor.env import ENV
from sqlglot.executor.table import RowReader, Table
from sqlglot.helper import csv_reader, ensure_list, subclasses
def scan_csv(self, step):
    alias = step.source.alias
    source = step.source.this
    with csv_reader(source) as reader:
        columns = next(reader)
        table = Table(columns)
        context = self.context({alias: table})
        yield context
        types = []
        for row in reader:
            if not types:
                for v in row:
                    try:
                        types.append(type(ast.literal_eval(v)))
                    except (ValueError, SyntaxError):
                        types.append(str)
            context.set_row(tuple((None if t is not str and v == '' else t(v) for t, v in zip(types, row))))
            yield context.table.reader