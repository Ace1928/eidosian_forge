import logging
import re
import warnings
from inspect import getmembers, ismethod
from webob import exc
from .secure import handle_security, cross_boundary
from .util import iscontroller, getargspec, _cfg
def find_object(obj, remainder, notfound_handlers, request):
    """
    'Walks' the url path in search of an action for which a controller is
    implemented and returns that controller object along with what's left
    of the remainder.
    """
    prev_obj = None
    while True:
        if obj is None:
            raise PecanNotFound
        if iscontroller(obj):
            if getattr(obj, 'custom_route', None) is None:
                return (obj, remainder)
        _detect_custom_path_segments(obj)
        if remainder:
            custom_route = __custom_routes__.get((obj.__class__, remainder[0]))
            if custom_route:
                return (getattr(obj, custom_route), remainder[1:])
        cross_boundary(prev_obj, obj)
        try:
            next_obj, rest = (remainder[0], remainder[1:])
            if next_obj == '':
                index = getattr(obj, 'index', None)
                if iscontroller(index):
                    return (index, rest)
        except IndexError:
            index = getattr(obj, 'index', None)
            if iscontroller(index):
                raise NonCanonicalPath(index, [])
        default = getattr(obj, '_default', None)
        if iscontroller(default):
            notfound_handlers.append(('_default', default, remainder))
        lookup = getattr(obj, '_lookup', None)
        if iscontroller(lookup):
            notfound_handlers.append(('_lookup', lookup, remainder))
        route = getattr(obj, '_route', None)
        if iscontroller(route):
            if len(getargspec(route).args) == 2:
                warnings.warn('The function signature for %s.%s._route is changing in the next version of pecan.\nPlease update to: `def _route(self, args, request)`.' % (obj.__class__.__module__, obj.__class__.__name__), DeprecationWarning)
                next_obj, next_remainder = route(remainder)
            else:
                next_obj, next_remainder = route(remainder, request)
            cross_boundary(route, next_obj)
            return (next_obj, next_remainder)
        if not remainder:
            raise PecanNotFound
        prev_remainder = remainder
        prev_obj = obj
        remainder = rest
        try:
            obj = getattr(obj, next_obj, None)
        except UnicodeEncodeError:
            obj = None
        if not obj and (not notfound_handlers) and hasattr(prev_obj, 'index'):
            if request.method in _cfg(prev_obj.index).get('generic_handlers', {}):
                return (prev_obj.index, prev_remainder)