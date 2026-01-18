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
def lookup_spawns_duplicates(opts, lookup_path):
    """
    Return True if the given lookup path spawns duplicates.
    """
    lookup_fields = lookup_path.split(LOOKUP_SEP)
    for field_name in lookup_fields:
        if field_name == 'pk':
            field_name = opts.pk.name
        try:
            field = opts.get_field(field_name)
        except FieldDoesNotExist:
            continue
        else:
            if hasattr(field, 'path_infos'):
                path_info = field.path_infos
                opts = path_info[-1].to_opts
                if any((path.m2m for path in path_info)):
                    return True
    return False