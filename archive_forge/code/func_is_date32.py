from pyarrow.lib import (is_boolean_value,  # noqa
import pyarrow.lib as lib
from pyarrow.util import doc
@doc(is_null, datatype='date32 (days)')
def is_date32(t):
    return t.id == lib.Type_DATE32