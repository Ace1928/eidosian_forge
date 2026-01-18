import functools
import gettext as gettext_module
import os
import re
import sys
import warnings
from asgiref.local import Local
from django.apps import apps
from django.conf import settings
from django.conf.locale import LANG_INFO
from django.core.exceptions import AppRegistryNotReady
from django.core.signals import setting_changed
from django.dispatch import receiver
from django.utils.regex_helper import _lazy_re_compile
from django.utils.safestring import SafeData, mark_safe
from . import to_language, to_locale
def parse_accept_lang_header(lang_string):
    """
    Parse the value of the Accept-Language header up to a maximum length.

    The value of the header is truncated to a maximum length to avoid potential
    denial of service and memory exhaustion attacks. Excessive memory could be
    used if the raw value is very large as it would be cached due to the use of
    functools.lru_cache() to avoid repetitive parsing of common header values.
    """
    if len(lang_string) <= ACCEPT_LANGUAGE_HEADER_MAX_LENGTH:
        return _parse_accept_lang_header(lang_string)
    if (index := lang_string.rfind(',', 0, ACCEPT_LANGUAGE_HEADER_MAX_LENGTH)) > 0:
        return _parse_accept_lang_header(lang_string[:index])
    return ()