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
def flatten_fieldsets(fieldsets):
    """Return a list of field names from an admin fieldsets structure."""
    field_names = []
    for name, opts in fieldsets:
        field_names.extend(flatten(opts['fields']))
    return field_names