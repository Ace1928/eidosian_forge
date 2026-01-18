from pyarrow.lib import (is_boolean_value,  # noqa
import pyarrow.lib as lib
from pyarrow.util import doc
@doc(is_null, datatype='time32')
def is_time32(t):
    return t.id == lib.Type_TIME32