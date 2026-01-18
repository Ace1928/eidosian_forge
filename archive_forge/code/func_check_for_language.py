from contextlib import ContextDecorator
from decimal import ROUND_UP, Decimal
from django.utils.autoreload import autoreload_started, file_changed
from django.utils.functional import lazy
from django.utils.regex_helper import _lazy_re_compile
def check_for_language(lang_code):
    return _trans.check_for_language(lang_code)