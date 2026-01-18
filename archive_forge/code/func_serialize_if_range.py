import re
from datetime import (
from collections import namedtuple
from webob.byterange import (
from webob.compat import (
from webob.datetime_utils import (
from webob.util import (
def serialize_if_range(value):
    if isinstance(value, (datetime, date)):
        return serialize_date(value)
    value = str(value)
    return value or None