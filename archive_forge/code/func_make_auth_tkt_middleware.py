import time as time_mod
from http.cookies import SimpleCookie
from urllib.parse import quote as url_quote
from urllib.parse import unquote as url_unquote
from paste import request
def make_auth_tkt_middleware(app, global_conf, secret=None, cookie_name='auth_tkt', secure=False, include_ip=True, logout_path=None):
    """
    Creates the `AuthTKTMiddleware
    <class-paste.auth.auth_tkt.AuthTKTMiddleware.html>`_.

    ``secret`` is required, but can be set globally or locally.
    """
    from paste.deploy.converters import asbool
    secure = asbool(secure)
    include_ip = asbool(include_ip)
    if secret is None:
        secret = global_conf.get('secret')
    if not secret:
        raise ValueError("You must provide a 'secret' (in global or local configuration)")
    return AuthTKTMiddleware(app, secret, cookie_name, secure, include_ip, logout_path or None)