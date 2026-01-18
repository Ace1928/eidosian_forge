from pyarrow.lib import (is_boolean_value,  # noqa
import pyarrow.lib as lib
from pyarrow.util import doc
@doc(is_null, datatype='date64 (milliseconds)')
def is_date64(t):
    return t.id == lib.Type_DATE64