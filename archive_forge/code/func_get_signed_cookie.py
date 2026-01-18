import codecs
import copy
from io import BytesIO
from itertools import chain
from urllib.parse import parse_qsl, quote, urlencode, urljoin, urlsplit
from django.conf import settings
from django.core import signing
from django.core.exceptions import (
from django.core.files import uploadhandler
from django.http.multipartparser import (
from django.utils.datastructures import (
from django.utils.encoding import escape_uri_path, iri_to_uri
from django.utils.functional import cached_property
from django.utils.http import is_same_domain, parse_header_parameters
from django.utils.regex_helper import _lazy_re_compile
def get_signed_cookie(self, key, default=RAISE_ERROR, salt='', max_age=None):
    """
        Attempt to return a signed cookie. If the signature fails or the
        cookie has expired, raise an exception, unless the `default` argument
        is provided,  in which case return that value.
        """
    try:
        cookie_value = self.COOKIES[key]
    except KeyError:
        if default is not RAISE_ERROR:
            return default
        else:
            raise
    try:
        value = signing.get_cookie_signer(salt=key + salt).unsign(cookie_value, max_age=max_age)
    except signing.BadSignature:
        if default is not RAISE_ERROR:
            return default
        else:
            raise
    return value