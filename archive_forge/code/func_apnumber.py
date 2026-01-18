import re
from datetime import date, datetime, timezone
from decimal import Decimal
from django import template
from django.template import defaultfilters
from django.utils.formats import number_format
from django.utils.safestring import mark_safe
from django.utils.timezone import is_aware
from django.utils.translation import gettext as _
from django.utils.translation import (
@register.filter(is_safe=True)
def apnumber(value):
    """
    For numbers 1-9, return the number spelled out. Otherwise, return the
    number. This follows Associated Press style.
    """
    try:
        value = int(value)
    except (TypeError, ValueError):
        return value
    if not 0 < value < 10:
        return value
    return (_('one'), _('two'), _('three'), _('four'), _('five'), _('six'), _('seven'), _('eight'), _('nine'))[value - 1]