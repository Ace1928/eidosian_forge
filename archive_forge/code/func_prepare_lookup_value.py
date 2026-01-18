import datetime
import decimal
import json
from collections import defaultdict
from functools import reduce
from operator import or_
from django.core.exceptions import FieldDoesNotExist
from django.db import models, router
from django.db.models.constants import LOOKUP_SEP
from django.db.models.deletion import Collector
from django.forms.utils import pretty_name
from django.urls import NoReverseMatch, reverse
from django.utils import formats, timezone
from django.utils.hashable import make_hashable
from django.utils.html import format_html
from django.utils.regex_helper import _lazy_re_compile
from django.utils.text import capfirst
from django.utils.translation import ngettext
from django.utils.translation import override as translation_override
def prepare_lookup_value(key, value, separator=','):
    """
    Return a lookup value prepared to be used in queryset filtering.
    """
    if isinstance(value, list):
        return [prepare_lookup_value(key, v, separator=separator) for v in value]
    if key.endswith('__in'):
        value = value.split(separator)
    elif key.endswith('__isnull'):
        value = value.lower() not in ('', 'false', '0')
    return value