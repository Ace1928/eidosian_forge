import os
import re
import sys
import time
from io import BytesIO
from typing import Callable, ClassVar, Dict, Iterator, List, Optional, Tuple
from urllib.parse import parse_qs
from wsgiref.simple_server import (
from dulwich import log_utils
from .protocol import ReceivableProtocol
from .repo import BaseRepo, NotGitRepository, Repo
from .server import (
def handle_service_request(req, backend, mat):
    service = mat.group().lstrip('/')
    logger.info('Handling service request for %s', service)
    handler_cls = req.handlers.get(service.encode('ascii'), None)
    if handler_cls is None:
        yield req.forbidden('Unsupported service')
        return
    try:
        get_repo(backend, mat)
    except NotGitRepository as e:
        yield req.not_found(str(e))
        return
    req.nocache()
    write = req.respond(HTTP_OK, 'application/x-%s-result' % service)
    if req.environ.get('HTTP_TRANSFER_ENCODING') == 'chunked':
        read = ChunkReader(req.environ['wsgi.input']).read
    else:
        read = req.environ['wsgi.input'].read
    proto = ReceivableProtocol(read, write)
    handler = handler_cls(backend, [url_prefix(mat)], proto, stateless_rpc=True)
    handler.handle()