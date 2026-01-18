from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple
import pyarrow as pa
from triad import Schema, assert_or_throw, to_uuid
from triad.utils.pyarrow import _type_to_expression
from triad.utils.schema import quote_name
from fugue.column.expressions import (
from fugue.column.functions import is_agg
from fugue.exceptions import FugueBug
def correct_select_schema(self, input_schema: Schema, select: SelectColumns, output_schema: Schema) -> Optional[Schema]:
    """Do partial schema inference from ``input_schema`` and ``select`` columns,
        then compare with the SQL output dataframe schema, and return the different
        part as a new schema, or None if there is no difference

        :param input_schema: input dataframe schema for the select statement
        :param select: the collection of select columns
        :param output_schema: schema of the output dataframe after executing the SQL
        :return: the difference as a new schema or None if no difference

        .. tip::

            This is particularly useful when the SQL engine messed up the schema of the
            output. For example, ``SELECT *`` should return a dataframe with the same
            schema of the input. However, for example a column ``a:int`` could become
            ``a:long`` in the output dataframe because of information loss. This
            function is designed to make corrections on column types when they can be
            inferred. This may not be perfect but it can solve major discrepancies.
        """
    cols = select.replace_wildcard(input_schema).assert_all_with_names()
    fields: List[pa.Field] = []
    for c in cols.all_cols:
        tp = c.infer_type(input_schema)
        if tp is not None and tp != output_schema[c.output_name].type:
            fields.append(pa.field(c.output_name, tp))
    if len(fields) == 0:
        return None
    return Schema(fields)