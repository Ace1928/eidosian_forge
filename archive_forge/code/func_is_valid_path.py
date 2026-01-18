from urllib.parse import unquote, urlsplit, urlunsplit
from asgiref.local import Local
from django.utils.functional import lazy
from django.utils.translation import override
from .exceptions import NoReverseMatch, Resolver404
from .resolvers import _get_cached_resolver, get_ns_resolver, get_resolver
from .utils import get_callable
def is_valid_path(path, urlconf=None):
    """
    Return the ResolverMatch if the given path resolves against the default URL
    resolver, False otherwise. This is a convenience method to make working
    with "is this a match?" cases easier, avoiding try...except blocks.
    """
    try:
        return resolve(path, urlconf)
    except Resolver404:
        return False