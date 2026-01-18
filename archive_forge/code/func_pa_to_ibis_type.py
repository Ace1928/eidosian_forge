from typing import Any, Callable, Dict, Optional, List
import ibis
import ibis.expr.datatypes as dt
import pyarrow as pa
from triad import Schema, extensible_class
from triad.utils.pyarrow import TRIAD_DEFAULT_TIMESTAMP
def pa_to_ibis_type(tp: pa.DataType) -> dt.DataType:
    if tp in _PYARROW_TO_IBIS:
        return _PYARROW_TO_IBIS[tp]
    if pa.types.is_timestamp(tp):
        if tp.tz is None:
            return dt.Timestamp()
        return dt.Timestamp(timezone=str(tp.tz))
    if pa.types.is_decimal(tp):
        return dt.Decimal(tp.precision, tp.scale)
    if pa.types.is_list(tp):
        ttp = pa_to_ibis_type(tp.value_type)
        return dt.Array(value_type=ttp)
    if pa.types.is_struct(tp):
        fields = [(f.name, pa_to_ibis_type(f.type)) for f in tp]
        return dt.Struct.from_tuples(fields)
    if pa.types.is_map(tp):
        return dt.Map(key_type=pa_to_ibis_type(tp.key_type), value_type=pa_to_ibis_type(tp.item_type))
    raise NotImplementedError(tp)