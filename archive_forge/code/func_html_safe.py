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
def html_safe(klass):
    """
    A decorator that defines the __html__ method. This helps non-Django
    templates to detect classes whose __str__ methods return SafeString.
    """
    if '__html__' in klass.__dict__:
        raise ValueError("can't apply @html_safe to %s because it defines __html__()." % klass.__name__)
    if '__str__' not in klass.__dict__:
        raise ValueError("can't apply @html_safe to %s because it doesn't define __str__()." % klass.__name__)
    klass_str = klass.__str__
    klass.__str__ = lambda self: mark_safe(klass_str(self))
    klass.__html__ = lambda self: str(self)
    return klass