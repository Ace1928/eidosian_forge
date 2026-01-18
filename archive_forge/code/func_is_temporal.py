from pyarrow.lib import (is_boolean_value,  # noqa
import pyarrow.lib as lib
from pyarrow.util import doc
@doc(is_null, datatype='date, time, timestamp or duration')
def is_temporal(t):
    return t.id in _TEMPORAL_TYPES