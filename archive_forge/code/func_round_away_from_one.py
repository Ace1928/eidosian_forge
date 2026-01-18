from contextlib import ContextDecorator
from decimal import ROUND_UP, Decimal
from django.utils.autoreload import autoreload_started, file_changed
from django.utils.functional import lazy
from django.utils.regex_helper import _lazy_re_compile
def round_away_from_one(value):
    return int(Decimal(value - 1).quantize(Decimal('0'), rounding=ROUND_UP)) + 1