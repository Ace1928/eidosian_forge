import datetime
import decimal
import functools
import re
import unicodedata
from importlib import import_module
from django.conf import settings
from django.utils import dateformat, numberformat
from django.utils.functional import lazy
from django.utils.translation import check_for_language, get_language, to_locale
def get_format_modules(lang=None):
    """Return a list of the format modules found."""
    if lang is None:
        lang = get_language()
    if lang not in _format_modules_cache:
        _format_modules_cache[lang] = list(iter_format_modules(lang, settings.FORMAT_MODULE_PATH))
    return _format_modules_cache[lang]