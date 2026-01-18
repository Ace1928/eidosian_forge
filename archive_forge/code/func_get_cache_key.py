import time
from collections import defaultdict
from hashlib import md5
from django.conf import settings
from django.core.cache import caches
from django.http import HttpResponse, HttpResponseNotModified
from django.utils.http import http_date, parse_etags, parse_http_date_safe, quote_etag
from django.utils.log import log_response
from django.utils.regex_helper import _lazy_re_compile
from django.utils.timezone import get_current_timezone_name
from django.utils.translation import get_language
def get_cache_key(request, key_prefix=None, method='GET', cache=None):
    """
    Return a cache key based on the request URL and query. It can be used
    in the request phase because it pulls the list of headers to take into
    account from the global URL registry and uses those to build a cache key
    to check against.

    If there isn't a headerlist stored, return None, indicating that the page
    needs to be rebuilt.
    """
    if key_prefix is None:
        key_prefix = settings.CACHE_MIDDLEWARE_KEY_PREFIX
    cache_key = _generate_cache_header_key(key_prefix, request)
    if cache is None:
        cache = caches[settings.CACHE_MIDDLEWARE_ALIAS]
    headerlist = cache.get(cache_key)
    if headerlist is not None:
        return _generate_cache_key(request, method, headerlist, key_prefix)
    else:
        return None