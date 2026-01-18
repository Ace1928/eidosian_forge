from urllib.parse import unquote, urlsplit, urlunsplit
from asgiref.local import Local
from django.utils.functional import lazy
from django.utils.translation import override
from .exceptions import NoReverseMatch, Resolver404
from .resolvers import _get_cached_resolver, get_ns_resolver, get_resolver
from .utils import get_callable
def clear_script_prefix():
    """
    Unset the script prefix for the current thread.
    """
    try:
        del _prefixes.value
    except AttributeError:
        pass