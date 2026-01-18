import mimetypes
from time import time as now
from email.utils import formatdate, parsedate_tz, mktime_tz
from urllib.request import AbstractDigestAuthHandler, parse_keqv_list, parse_http_list
from .httpexceptions import HTTPBadRequest
def normalize_headers(response_headers, strict=True):
    """
    sort headers as suggested by  RFC 2616

    This alters the underlying response_headers to use the common
    name for each header; as well as sorting them with general
    headers first, followed by request/response headers, then
    entity headers, and unknown headers last.
    """
    category = {}
    for idx in range(len(response_headers)):
        key, val = response_headers[idx]
        head = get_header(key, strict)
        if not head:
            newhead = '-'.join([x.capitalize() for x in key.replace('_', '-').split('-')])
            response_headers[idx] = (newhead, val)
            category[newhead] = 4
            continue
        response_headers[idx] = (str(head), val)
        category[str(head)] = head.sort_order

    def key_func(item):
        value = item[0]
        return (category[value], value)
    response_headers.sort(key=key_func)