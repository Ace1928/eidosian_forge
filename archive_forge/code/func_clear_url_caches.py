from urllib.parse import unquote, urlsplit, urlunsplit
from asgiref.local import Local
from django.utils.functional import lazy
from django.utils.translation import override
from .exceptions import NoReverseMatch, Resolver404
from .resolvers import _get_cached_resolver, get_ns_resolver, get_resolver
from .utils import get_callable
def clear_url_caches():
    get_callable.cache_clear()
    _get_cached_resolver.cache_clear()
    get_ns_resolver.cache_clear()