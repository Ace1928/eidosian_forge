from calendar import timegm
from decimal import Decimal as MyDecimal, ROUND_HALF_EVEN
from email.utils import formatdate
import six
from flask_restful import marshal
from flask import url_for, request
def is_indexable_but_not_string(obj):
    return not hasattr(obj, 'strip') and hasattr(obj, '__iter__')