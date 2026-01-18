import logging
import sys
import weakref
import webob
from wsme.exc import ClientSideError, UnknownFunction
from wsme.protocol import getprotocol
from wsme.rest import scan_api
import wsme.api
import wsme.types
def default_prepare_response_body(request, results):
    r = None
    sep = None
    for value in results:
        if sep is None:
            if isinstance(value, str):
                sep = '\n'
                r = ''
            else:
                sep = b'\n'
                r = b''
        else:
            r += sep
        r += value
    return r