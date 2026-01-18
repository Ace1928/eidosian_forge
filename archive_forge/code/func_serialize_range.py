import re
from datetime import (
from collections import namedtuple
from webob.byterange import (
from webob.compat import (
from webob.datetime_utils import (
from webob.util import (
def serialize_range(value):
    if not value:
        return None
    elif isinstance(value, (list, tuple)):
        return str(Range(*value))
    else:
        assert isinstance(value, str)
        return value