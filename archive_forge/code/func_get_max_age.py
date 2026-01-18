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
def get_max_age(response):
    """
    Return the max-age from the response Cache-Control header as an integer,
    or None if it wasn't found or wasn't an integer.
    """
    if not response.has_header('Cache-Control'):
        return
    cc = dict((_to_tuple(el) for el in cc_delim_re.split(response.headers['Cache-Control'])))
    try:
        return int(cc['max-age'])
    except (ValueError, TypeError, KeyError):
        pass