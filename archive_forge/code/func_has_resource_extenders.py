import collections
import inspect
from oslo_log import log as logging
from oslo_utils import timeutils
from neutron_lib.utils import helpers
def has_resource_extenders(klass):
    """Decorator to setup __new__ method in classes to extend resources.

    Any method decorated with @extends above is an unbound method on a class.
    This decorator sets up the class __new__ method to add the bound
    method to _resource_extend_functions after object instantiation.
    """
    orig_new = klass.__new__
    new_inherited = '__new__' not in klass.__dict__

    @staticmethod
    def replacement_new(cls, *args, **kwargs):
        if new_inherited:
            super_new = super(klass, cls).__new__
            if super_new is object.__new__:
                instance = super_new(cls)
            else:
                instance = super_new(cls, *args, **kwargs)
        else:
            instance = orig_new(cls, *args, **kwargs)
        if getattr(instance, _DECORATED_METHODS_REGISTERED, False):
            return instance
        for name, unbound_method in inspect.getmembers(cls):
            if not inspect.ismethod(unbound_method) and (not inspect.isfunction(unbound_method)):
                continue
            method = getattr(unbound_method, 'im_func', unbound_method)
            if method not in _DECORATED_EXTEND_METHODS:
                continue
            for resource in _DECORATED_EXTEND_METHODS[method]:
                register_funcs(resource, [method])
        setattr(instance, _DECORATED_METHODS_REGISTERED, True)
        return instance
    klass.__new__ = replacement_new
    return klass