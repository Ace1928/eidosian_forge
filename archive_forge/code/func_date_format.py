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
def date_format(value, format=None, use_l10n=None):
    """
    Format a datetime.date or datetime.datetime object using a
    localizable format.

    If use_l10n is provided and is not None, that will force the value to
    be localized (or not), otherwise it's always localized.
    """
    return dateformat.format(value, get_format(format or 'DATE_FORMAT', use_l10n=use_l10n))