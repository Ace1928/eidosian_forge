import re
from datetime import (
from collections import namedtuple
from webob.byterange import (
from webob.compat import (
from webob.datetime_utils import (
from webob.util import (
def parse_int_safe(value):
    if value is None or value == '':
        return None
    try:
        return int(value)
    except ValueError:
        return None