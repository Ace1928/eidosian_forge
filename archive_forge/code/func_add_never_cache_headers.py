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
def add_never_cache_headers(response):
    """
    Add headers to a response to indicate that a page should never be cached.
    """
    patch_response_headers(response, cache_timeout=-1)
    patch_cache_control(response, no_cache=True, no_store=True, must_revalidate=True, private=True)