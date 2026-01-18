import os
import re
import six
from six.moves import urllib
from routes import request_config
def as_unicode(value, encoding, errors='strict'):
    if value is not None and isinstance(value, bytes):
        return value.decode(encoding, errors)
    return value