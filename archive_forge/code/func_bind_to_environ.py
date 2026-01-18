from __future__ import annotations
import typing as t
import warnings
from pprint import pformat
from threading import Lock
from urllib.parse import quote
from urllib.parse import urljoin
from urllib.parse import urlunsplit
from .._internal import _get_environ
from .._internal import _wsgi_decoding_dance
from ..datastructures import ImmutableDict
from ..datastructures import MultiDict
from ..exceptions import BadHost
from ..exceptions import HTTPException
from ..exceptions import MethodNotAllowed
from ..exceptions import NotFound
from ..urls import _urlencode
from ..wsgi import get_host
from .converters import DEFAULT_CONVERTERS
from .exceptions import BuildError
from .exceptions import NoMatch
from .exceptions import RequestAliasRedirect
from .exceptions import RequestPath
from .exceptions import RequestRedirect
from .exceptions import WebsocketMismatch
from .matcher import StateMachineMatcher
from .rules import _simple_rule_re
from .rules import Rule
def bind_to_environ(self, environ: WSGIEnvironment | Request, server_name: str | None=None, subdomain: str | None=None) -> MapAdapter:
    """Like :meth:`bind` but you can pass it an WSGI environment and it
        will fetch the information from that dictionary.  Note that because of
        limitations in the protocol there is no way to get the current
        subdomain and real `server_name` from the environment.  If you don't
        provide it, Werkzeug will use `SERVER_NAME` and `SERVER_PORT` (or
        `HTTP_HOST` if provided) as used `server_name` with disabled subdomain
        feature.

        If `subdomain` is `None` but an environment and a server name is
        provided it will calculate the current subdomain automatically.
        Example: `server_name` is ``'example.com'`` and the `SERVER_NAME`
        in the wsgi `environ` is ``'staging.dev.example.com'`` the calculated
        subdomain will be ``'staging.dev'``.

        If the object passed as environ has an environ attribute, the value of
        this attribute is used instead.  This allows you to pass request
        objects.  Additionally `PATH_INFO` added as a default of the
        :class:`MapAdapter` so that you don't have to pass the path info to
        the match method.

        .. versionchanged:: 1.0.0
            If the passed server name specifies port 443, it will match
            if the incoming scheme is ``https`` without a port.

        .. versionchanged:: 1.0.0
            A warning is shown when the passed server name does not
            match the incoming WSGI server name.

        .. versionchanged:: 0.8
           This will no longer raise a ValueError when an unexpected server
           name was passed.

        .. versionchanged:: 0.5
            previously this method accepted a bogus `calculate_subdomain`
            parameter that did not have any effect.  It was removed because
            of that.

        :param environ: a WSGI environment.
        :param server_name: an optional server name hint (see above).
        :param subdomain: optionally the current subdomain (see above).
        """
    env = _get_environ(environ)
    wsgi_server_name = get_host(env).lower()
    scheme = env['wsgi.url_scheme']
    upgrade = any((v.strip() == 'upgrade' for v in env.get('HTTP_CONNECTION', '').lower().split(',')))
    if upgrade and env.get('HTTP_UPGRADE', '').lower() == 'websocket':
        scheme = 'wss' if scheme == 'https' else 'ws'
    if server_name is None:
        server_name = wsgi_server_name
    else:
        server_name = server_name.lower()
        if scheme in {'http', 'ws'} and server_name.endswith(':80'):
            server_name = server_name[:-3]
        elif scheme in {'https', 'wss'} and server_name.endswith(':443'):
            server_name = server_name[:-4]
    if subdomain is None and (not self.host_matching):
        cur_server_name = wsgi_server_name.split('.')
        real_server_name = server_name.split('.')
        offset = -len(real_server_name)
        if cur_server_name[offset:] != real_server_name:
            warnings.warn(f"Current server name {wsgi_server_name!r} doesn't match configured server name {server_name!r}", stacklevel=2)
            subdomain = '<invalid>'
        else:
            subdomain = '.'.join(filter(None, cur_server_name[:offset]))

    def _get_wsgi_string(name: str) -> str | None:
        val = env.get(name)
        if val is not None:
            return _wsgi_decoding_dance(val)
        return None
    script_name = _get_wsgi_string('SCRIPT_NAME')
    path_info = _get_wsgi_string('PATH_INFO')
    query_args = _get_wsgi_string('QUERY_STRING')
    return Map.bind(self, server_name, script_name, subdomain, scheme, env['REQUEST_METHOD'], path_info, query_args=query_args)