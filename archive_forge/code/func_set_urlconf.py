from urllib.parse import unquote, urlsplit, urlunsplit
from asgiref.local import Local
from django.utils.functional import lazy
from django.utils.translation import override
from .exceptions import NoReverseMatch, Resolver404
from .resolvers import _get_cached_resolver, get_ns_resolver, get_resolver
from .utils import get_callable
def set_urlconf(urlconf_name):
    """
    Set the URLconf for the current thread (overriding the default one in
    settings). If urlconf_name is None, revert back to the default.
    """
    if urlconf_name:
        _urlconfs.value = urlconf_name
    elif hasattr(_urlconfs, 'value'):
        del _urlconfs.value