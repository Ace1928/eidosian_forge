import re
from datetime import (
from collections import namedtuple
from webob.byterange import (
from webob.compat import (
from webob.datetime_utils import (
from webob.util import (
def list_header(header, rfc_section):
    prop = header_getter(header, rfc_section)
    return converter(prop, parse_list, serialize_list, 'list')