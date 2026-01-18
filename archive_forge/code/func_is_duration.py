from pyarrow.lib import (is_boolean_value,  # noqa
import pyarrow.lib as lib
from pyarrow.util import doc
@doc(is_null, datatype='duration')
def is_duration(t):
    return t.id == lib.Type_DURATION