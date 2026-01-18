from __future__ import annotations
import email.utils
import re
import typing as t
import warnings
from datetime import date
from datetime import datetime
from datetime import time
from datetime import timedelta
from datetime import timezone
from enum import Enum
from hashlib import sha1
from time import mktime
from time import struct_time
from urllib.parse import quote
from urllib.parse import unquote
from urllib.request import parse_http_list as _parse_list_header
from ._internal import _dt_as_utc
from ._internal import _plain_int
from . import datastructures as ds
from .sansio import http as _sansio_http
def dump_cookie(key: str, value: str='', max_age: timedelta | int | None=None, expires: str | datetime | int | float | None=None, path: str | None='/', domain: str | None=None, secure: bool=False, httponly: bool=False, sync_expires: bool=True, max_size: int=4093, samesite: str | None=None) -> str:
    """Create a Set-Cookie header without the ``Set-Cookie`` prefix.

    The return value is usually restricted to ascii as the vast majority
    of values are properly escaped, but that is no guarantee. It's
    tunneled through latin1 as required by :pep:`3333`.

    The return value is not ASCII safe if the key contains unicode
    characters.  This is technically against the specification but
    happens in the wild.  It's strongly recommended to not use
    non-ASCII values for the keys.

    :param max_age: should be a number of seconds, or `None` (default) if
                    the cookie should last only as long as the client's
                    browser session.  Additionally `timedelta` objects
                    are accepted, too.
    :param expires: should be a `datetime` object or unix timestamp.
    :param path: limits the cookie to a given path, per default it will
                 span the whole domain.
    :param domain: Use this if you want to set a cross-domain cookie. For
                   example, ``domain="example.com"`` will set a cookie
                   that is readable by the domain ``www.example.com``,
                   ``foo.example.com`` etc. Otherwise, a cookie will only
                   be readable by the domain that set it.
    :param secure: The cookie will only be available via HTTPS
    :param httponly: disallow JavaScript to access the cookie.  This is an
                     extension to the cookie standard and probably not
                     supported by all browsers.
    :param charset: the encoding for string values.
    :param sync_expires: automatically set expires if max_age is defined
                         but expires not.
    :param max_size: Warn if the final header value exceeds this size. The
        default, 4093, should be safely `supported by most browsers
        <cookie_>`_. Set to 0 to disable this check.
    :param samesite: Limits the scope of the cookie such that it will
        only be attached to requests if those requests are same-site.

    .. _`cookie`: http://browsercookielimits.squawky.net/

    .. versionchanged:: 3.0
        Passing bytes, and the ``charset`` parameter, were removed.

    .. versionchanged:: 2.3.3
        The ``path`` parameter is ``/`` by default.

    .. versionchanged:: 2.3.1
        The value allows more characters without quoting.

    .. versionchanged:: 2.3
        ``localhost`` and other names without a dot are allowed for the domain. A
        leading dot is ignored.

    .. versionchanged:: 2.3
        The ``path`` parameter is ``None`` by default.

    .. versionchanged:: 1.0.0
        The string ``'None'`` is accepted for ``samesite``.
    """
    if path is not None:
        path = quote(path, safe="%!$&'()*+,/:=@")
    if domain:
        domain = domain.partition(':')[0].lstrip('.').encode('idna').decode('ascii')
    if isinstance(max_age, timedelta):
        max_age = int(max_age.total_seconds())
    if expires is not None:
        if not isinstance(expires, str):
            expires = http_date(expires)
    elif max_age is not None and sync_expires:
        expires = http_date(datetime.now(tz=timezone.utc).timestamp() + max_age)
    if samesite is not None:
        samesite = samesite.title()
        if samesite not in {'Strict', 'Lax', 'None'}:
            raise ValueError("SameSite must be 'Strict', 'Lax', or 'None'.")
    if not _cookie_no_quote_re.fullmatch(value):
        value = _cookie_slash_re.sub(lambda m: _cookie_slash_map[m.group()], value.encode()).decode('ascii')
        value = f'"{value}"'
    buf = [f'{key.encode().decode('latin1')}={value}']
    for k, v in (('Domain', domain), ('Expires', expires), ('Max-Age', max_age), ('Secure', secure), ('HttpOnly', httponly), ('Path', path), ('SameSite', samesite)):
        if v is None or v is False:
            continue
        if v is True:
            buf.append(k)
            continue
        buf.append(f'{k}={v}')
    rv = '; '.join(buf)
    cookie_size = len(rv)
    if max_size and cookie_size > max_size:
        value_size = len(value)
        warnings.warn(f"The '{key}' cookie is too large: the value was {value_size} bytes but the header required {cookie_size - value_size} extra bytes. The final size was {cookie_size} bytes but the limit is {max_size} bytes. Browsers may silently ignore cookies larger than this.", stacklevel=2)
    return rv