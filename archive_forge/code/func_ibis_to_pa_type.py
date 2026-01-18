from typing import Any, Callable, Dict, Optional, List
import ibis
import ibis.expr.datatypes as dt
import pyarrow as pa
from triad import Schema, extensible_class
from triad.utils.pyarrow import TRIAD_DEFAULT_TIMESTAMP
def ibis_to_pa_type(tp: dt.DataType) -> pa.DataType:
    if tp in _IBIS_TO_PYARROW:
        return _IBIS_TO_PYARROW[tp]
    if isinstance(tp, dt.Timestamp):
        if tp.timezone is None:
            return TRIAD_DEFAULT_TIMESTAMP
        else:
            return pa.timestamp('us', tp.timezone)
    if isinstance(tp, dt.Decimal) and tp.precision is not None:
        return pa.decimal128(tp.precision, 0 if tp.scale is None else tp.scale)
    if isinstance(tp, dt.Array):
        ttp = ibis_to_pa_type(tp.value_type)
        return pa.list_(ttp)
    if isinstance(tp, dt.Struct):
        fields = [pa.field(n, ibis_to_pa_type(t)) for n, t in zip(tp.names, tp.types)]
        return pa.struct(fields)
    if isinstance(tp, dt.Map):
        return pa.map_(ibis_to_pa_type(tp.key_type), ibis_to_pa_type(tp.value_type))
    raise NotImplementedError(tp)