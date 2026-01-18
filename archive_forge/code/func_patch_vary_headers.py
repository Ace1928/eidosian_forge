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
def patch_vary_headers(response, newheaders):
    """
    Add (or update) the "Vary" header in the given HttpResponse object.
    newheaders is a list of header names that should be in "Vary". If headers
    contains an asterisk, then "Vary" header will consist of a single asterisk
    '*'. Otherwise, existing headers in "Vary" aren't removed.
    """
    if response.has_header('Vary'):
        vary_headers = cc_delim_re.split(response.headers['Vary'])
    else:
        vary_headers = []
    existing_headers = {header.lower() for header in vary_headers}
    additional_headers = [newheader for newheader in newheaders if newheader.lower() not in existing_headers]
    vary_headers += additional_headers
    if '*' in vary_headers:
        response.headers['Vary'] = '*'
    else:
        response.headers['Vary'] = ', '.join(vary_headers)