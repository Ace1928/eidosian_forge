from typing import Any, Dict, Iterable, List, Optional, Union
import pyarrow as pa
from triad import Schema, assert_or_throw, to_uuid
from triad.utils.pyarrow import _type_to_expression, to_pa_datatype
@property
def body_str(self) -> str:

    def to_str(v: Any):
        if isinstance(v, str):
            return f"'{v}'"
        if isinstance(v, bool):
            return 'TRUE' if v else 'FALSE'
        return str(v)
    a1 = [to_str(x) for x in self.args]
    a2 = [k + '=' + to_str(v) for k, v in self.kwargs.items()]
    args = ','.join(a1 + a2)
    distinct = 'DISTINCT ' if self.is_distinct else ''
    return f'{self.func}({distinct}{args})'