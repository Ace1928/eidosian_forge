from pyarrow.lib import (is_boolean_value,  # noqa
import pyarrow.lib as lib
from pyarrow.util import doc
@doc(is_null, datatype='run-end encoded')
def is_run_end_encoded(t):
    return t.id == lib.Type_RUN_END_ENCODED