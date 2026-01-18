import re
from datetime import (
from collections import namedtuple
from webob.byterange import (
from webob.compat import (
from webob.datetime_utils import (
from webob.util import (
def converter_date(prop):
    return converter(prop, parse_date, serialize_date, 'HTTP date')