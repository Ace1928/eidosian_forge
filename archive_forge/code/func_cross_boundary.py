from functools import wraps
from inspect import getmembers, isfunction
from webob import exc
from .compat import is_bound_method as ismethod
from .decorators import expose
from .util import _cfg, iscontroller
def cross_boundary(prev_obj, obj):
    """ Check permissions as we move between object instances. """
    if prev_obj is None:
        return
    if isinstance(obj, _SecuredAttribute):
        obj.parent = prev_obj
    if hasattr(prev_obj, '_pecan'):
        if obj not in prev_obj._pecan.get('unlocked', []):
            handle_security(prev_obj)