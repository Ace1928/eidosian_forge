from pyarrow.lib import (is_boolean_value,  # noqa
import pyarrow.lib as lib
from pyarrow.util import doc
@doc(is_unicode, method='is_large_string')
def is_large_unicode(t):
    return is_large_string(t)