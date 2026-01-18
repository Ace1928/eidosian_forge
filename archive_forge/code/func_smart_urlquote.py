import html
import json
import re
import warnings
from html.parser import HTMLParser
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit, urlunsplit
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.encoding import punycode
from django.utils.functional import Promise, keep_lazy, keep_lazy_text
from django.utils.http import RFC3986_GENDELIMS, RFC3986_SUBDELIMS
from django.utils.regex_helper import _lazy_re_compile
from django.utils.safestring import SafeData, SafeString, mark_safe
from django.utils.text import normalize_newlines
def smart_urlquote(url):
    """Quote a URL if it isn't already quoted."""

    def unquote_quote(segment):
        segment = unquote(segment)
        return quote(segment, safe=RFC3986_SUBDELIMS + RFC3986_GENDELIMS + '~')
    try:
        scheme, netloc, path, query, fragment = urlsplit(url)
    except ValueError:
        return unquote_quote(url)
    try:
        netloc = punycode(netloc)
    except UnicodeError:
        return unquote_quote(url)
    if query:
        query_parts = [(unquote(q[0]), unquote(q[1])) for q in parse_qsl(query, keep_blank_values=True)]
        query = urlencode(query_parts)
    path = unquote_quote(path)
    fragment = unquote_quote(fragment)
    return urlunsplit((scheme, netloc, path, query, fragment))