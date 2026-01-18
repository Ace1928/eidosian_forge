from __future__ import annotations
import json
import urllib.parse as urlparse
from typing import (
import param
from ..models.location import Location as _BkLocation
from ..reactive import Syncable
from ..util import edit_readonly, parse_query
from .document import create_doc_if_none_exists
from .state import state
@classmethod
def from_request(cls, request):
    try:
        from bokeh.server.contexts import _RequestProxy
        if not isinstance(request, _RequestProxy) or request._request is None:
            return cls()
    except ImportError:
        return cls()
    params = {}
    href = ''
    if request.protocol:
        params['protocol'] = href = f'{request.protocol}:'
    if request.host:
        href += f'//{request.host}'
        if ':' in request.host:
            params['hostname'], params['port'] = request.host.split(':')
        else:
            params['hostname'] = request.host
    if request.uri:
        search = hash = None
        href += request.uri
        if '?' in request.uri and '#' in request.uri:
            params['pathname'], query = request.uri.split('?')
            search, hash = query.split('#')
        elif '?' in request.uri:
            params['pathname'], search = request.uri.split('?')
        elif '#' in request.uri:
            params['pathname'], hash = request.uri.split('#')
        else:
            params['pathname'] = request.uri
        if search:
            params['search'] = f'?{search}'
        if hash:
            params['hash'] = f'#{hash}'
    params['href'] = href
    loc = cls()
    with edit_readonly(loc):
        loc.param.update(params)
    return loc