import logging
import re
import warnings
from inspect import getmembers, ismethod
from webob import exc
from .secure import handle_security, cross_boundary
from .util import iscontroller, getargspec, _cfg
def lookup_controller(obj, remainder, request=None):
    """
    Traverses the requested url path and returns the appropriate controller
    object, including default routes.

    Handles common errors gracefully.
    """
    if request is None:
        warnings.warn('The function signature for %s.lookup_controller is changing in the next version of pecan.\nPlease update to: `lookup_controller(self, obj, remainder, request)`.' % (__name__,), DeprecationWarning)
    notfound_handlers = []
    while True:
        try:
            obj, remainder = find_object(obj, remainder, notfound_handlers, request)
            handle_security(obj)
            return (obj, remainder)
        except (exc.HTTPNotFound, exc.HTTPMethodNotAllowed, PecanNotFound) as e:
            if isinstance(e, PecanNotFound):
                e = exc.HTTPNotFound()
            while notfound_handlers:
                name, obj, remainder = notfound_handlers.pop()
                if name == '_default':
                    return (obj, remainder)
                else:
                    result = handle_lookup_traversal(obj, remainder)
                    if result:
                        if remainder == [''] and len(obj._pecan['argspec'].args) > 1:
                            raise e
                        obj_, remainder_ = result
                        return lookup_controller(obj_, remainder_, request)
            else:
                raise e