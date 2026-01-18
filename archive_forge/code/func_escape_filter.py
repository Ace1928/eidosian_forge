import random as random_module
import re
import types
import warnings
from decimal import ROUND_HALF_UP, Context, Decimal, InvalidOperation, getcontext
from functools import wraps
from inspect import unwrap
from operator import itemgetter
from pprint import pformat
from urllib.parse import quote
from django.utils import formats
from django.utils.dateformat import format, time_format
from django.utils.deprecation import RemovedInDjango51Warning
from django.utils.encoding import iri_to_uri
from django.utils.html import avoid_wrapping, conditional_escape, escape, escapejs
from django.utils.html import json_script as _json_script
from django.utils.html import linebreaks, strip_tags
from django.utils.html import urlize as _urlize
from django.utils.safestring import SafeData, mark_safe
from django.utils.text import Truncator, normalize_newlines, phone2numeric
from django.utils.text import slugify as _slugify
from django.utils.text import wrap
from django.utils.timesince import timesince, timeuntil
from django.utils.translation import gettext, ngettext
from .base import VARIABLE_ATTRIBUTE_SEPARATOR
from .library import Library
@register.filter('escape', is_safe=True)
@stringfilter
def escape_filter(value):
    """Mark the value as a string that should be auto-escaped."""
    return conditional_escape(value)