from contextlib import ContextDecorator
from decimal import ROUND_UP, Decimal
from django.utils.autoreload import autoreload_started, file_changed
from django.utils.functional import lazy
from django.utils.regex_helper import _lazy_re_compile
def ngettext_lazy(singular, plural, number=None):
    return lazy_number(ngettext, str, singular=singular, plural=plural, number=number)