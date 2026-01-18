from __future__ import annotations
import typing as t
from types import TracebackType
from urllib.parse import urlparse
from warnings import warn
from ..datastructures import Headers
from ..http import is_entity_header
from ..wsgi import FileWrapper
def check_environ(self, environ: WSGIEnvironment) -> None:
    if type(environ) is not dict:
        warn('WSGI environment is not a standard Python dict.', WSGIWarning, stacklevel=4)
    for key in ('REQUEST_METHOD', 'SERVER_NAME', 'SERVER_PORT', 'wsgi.version', 'wsgi.input', 'wsgi.errors', 'wsgi.multithread', 'wsgi.multiprocess', 'wsgi.run_once'):
        if key not in environ:
            warn(f'Required environment key {key!r} not found', WSGIWarning, stacklevel=3)
    if environ['wsgi.version'] != (1, 0):
        warn('Environ is not a WSGI 1.0 environ.', WSGIWarning, stacklevel=3)
    script_name = environ.get('SCRIPT_NAME', '')
    path_info = environ.get('PATH_INFO', '')
    if script_name and script_name[0] != '/':
        warn(f"'SCRIPT_NAME' does not start with a slash: {script_name!r}", WSGIWarning, stacklevel=3)
    if path_info and path_info[0] != '/':
        warn(f"'PATH_INFO' does not start with a slash: {path_info!r}", WSGIWarning, stacklevel=3)