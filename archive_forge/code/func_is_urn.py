from __future__ import (absolute_import, division, print_function)
from urllib.parse import urlparse
def is_urn(value):
    return is_uri(value, ['urn'])