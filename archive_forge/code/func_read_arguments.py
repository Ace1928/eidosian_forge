import os.path
import logging
from wsme.utils import OrderedDict
from wsme.protocol import CallContext, Protocol, media_type_accept
import wsme.rest
from wsme.rest import json
from wsme.rest import xml
import wsme.runtime
def read_arguments(self, context):
    request = context.request
    funcdef = context.funcdef
    body = None
    if request.content_length not in (None, 0, '0'):
        body = request.body
    if not body and '__body__' in request.params:
        body = request.params['__body__']
    args, kwargs = wsme.rest.args.combine_args(funcdef, (wsme.rest.args.args_from_params(funcdef, request.params), wsme.rest.args.args_from_body(funcdef, body, context.inmime)))
    wsme.runtime.check_arguments(funcdef, args, kwargs)
    return kwargs