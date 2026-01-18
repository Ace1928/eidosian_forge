import functools
import logging
import sys
import inspect
import flask
import wsme
import wsme.api
import wsme.rest.args
import wsme.rest.json
import wsme.rest.xml
from wsme.utils import is_valid_code
def get_dataformat():
    if 'Accept' in flask.request.headers:
        for t in TYPES:
            if t in flask.request.headers['Accept']:
                return TYPES[t]
    req_dataformat = getattr(flask.request, 'response_type', None)
    if req_dataformat in TYPES:
        return TYPES[req_dataformat]
    log.info('Could not determine what format is wanted by the caller, falling back to JSON')
    return wsme.rest.json