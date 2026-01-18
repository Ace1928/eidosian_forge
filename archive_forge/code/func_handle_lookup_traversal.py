import logging
import re
import warnings
from inspect import getmembers, ismethod
from webob import exc
from .secure import handle_security, cross_boundary
from .util import iscontroller, getargspec, _cfg
def handle_lookup_traversal(obj, args):
    try:
        result = obj(*args)
    except TypeError as te:
        logger.debug('Got exception calling lookup(): %s (%s)', te, te.args)
    else:
        if result:
            prev_obj = obj
            obj, remainder = result
            cross_boundary(prev_obj, obj)
            return result