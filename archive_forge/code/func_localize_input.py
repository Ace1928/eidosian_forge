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
def localize_input(value, default=None):
    """
    Check if an input value is a localizable type and return it
    formatted with the appropriate formatting string of the current locale.
    """
    if isinstance(value, str):
        return value
    elif isinstance(value, bool):
        return str(value)
    elif isinstance(value, (decimal.Decimal, float, int)):
        return number_format(value)
    elif isinstance(value, datetime.datetime):
        format = default or get_format('DATETIME_INPUT_FORMATS')[0]
        format = sanitize_strftime_format(format)
        return value.strftime(format)
    elif isinstance(value, datetime.date):
        format = default or get_format('DATE_INPUT_FORMATS')[0]
        format = sanitize_strftime_format(format)
        return value.strftime(format)
    elif isinstance(value, datetime.time):
        format = default or get_format('TIME_INPUT_FORMATS')[0]
        return value.strftime(format)
    return value