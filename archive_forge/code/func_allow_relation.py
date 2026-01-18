import pkgutil
from importlib import import_module
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.utils.connection import ConnectionDoesNotExist  # NOQA: F401
from django.utils.connection import BaseConnectionHandler
from django.utils.functional import cached_property
from django.utils.module_loading import import_string
def allow_relation(self, obj1, obj2, **hints):
    for router in self.routers:
        try:
            method = router.allow_relation
        except AttributeError:
            pass
        else:
            allow = method(obj1, obj2, **hints)
            if allow is not None:
                return allow
    return obj1._state.db == obj2._state.db