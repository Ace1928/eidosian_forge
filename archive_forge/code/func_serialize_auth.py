import re
from datetime import (
from collections import namedtuple
from webob.byterange import (
from webob.compat import (
from webob.datetime_utils import (
from webob.util import (
def serialize_auth(val):
    if isinstance(val, (tuple, list)):
        authtype, params = val
        if isinstance(params, dict):
            params = ', '.join(map('%s="%s"'.__mod__, params.items()))
        assert isinstance(params, str)
        return '%s %s' % (authtype, params)
    return val