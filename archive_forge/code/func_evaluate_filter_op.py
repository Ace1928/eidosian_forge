import errno
from eventlet.green import socket
import functools
import os
import re
import urllib
import glance_store
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import netutils
from oslo_utils import strutils
from webob import exc
from glance.common import exception
from glance.common import location_strategy
from glance.common import timeutils
from glance.common import wsgi
from glance.i18n import _, _LE, _LW
def evaluate_filter_op(value, operator, threshold):
    """Evaluate a comparison operator.
    Designed for use on a comparative-filtering query field.

    :param value: evaluated against the operator, as left side of expression
    :param operator: any supported filter operation
    :param threshold: to compare value against, as right side of expression

    :raises InvalidFilterOperatorValue: if an unknown operator is provided

    :returns: boolean result of applied comparison

    """
    if operator == 'gt':
        return value > threshold
    elif operator == 'gte':
        return value >= threshold
    elif operator == 'lt':
        return value < threshold
    elif operator == 'lte':
        return value <= threshold
    elif operator == 'neq':
        return value != threshold
    elif operator == 'eq':
        return value == threshold
    msg = _('Unable to filter on a unknown operator.')
    raise exception.InvalidFilterOperatorValue(msg)