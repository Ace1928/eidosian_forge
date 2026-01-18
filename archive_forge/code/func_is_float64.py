from pyarrow.lib import (is_boolean_value,  # noqa
import pyarrow.lib as lib
from pyarrow.util import doc
@doc(is_null, datatype='float64 (double precision)')
def is_float64(t):
    return t.id == lib.Type_DOUBLE