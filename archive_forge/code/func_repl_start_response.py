import os
import re
from paste.fileapp import FileApp
from paste.response import header_value, remove_header
def repl_start_response(status, headers, exc_info=None):
    ct = header_value(headers, 'content-type')
    if ct and ct.startswith('text/html'):
        type.append(ct)
        remove_header(headers, 'content-length')
        start_response(status, headers, exc_info)
        return body.append
    return start_response(status, headers, exc_info)