from __future__ import annotations
from collections.abc import Mapping
from functools import partial
from typing import NamedTuple
from urllib.parse import parse_qsl, quote, unquote, urlparse
from ..log import get_logger
def parse_ssl_cert_reqs(query_value):
    """Given the query parameter for ssl_cert_reqs, return the SSL constant or None."""
    if ssl_available:
        query_value_to_constant = {'CERT_REQUIRED': ssl.CERT_REQUIRED, 'CERT_OPTIONAL': ssl.CERT_OPTIONAL, 'CERT_NONE': ssl.CERT_NONE, 'required': ssl.CERT_REQUIRED, 'optional': ssl.CERT_OPTIONAL, 'none': ssl.CERT_NONE}
        return query_value_to_constant[query_value]
    else:
        return None