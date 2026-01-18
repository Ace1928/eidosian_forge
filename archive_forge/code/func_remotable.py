import abc
import collections
from collections import abc as collections_abc
import copy
import functools
import logging
import warnings
import oslo_messaging as messaging
from oslo_utils import excutils
from oslo_utils import versionutils as vutils
from oslo_versionedobjects._i18n import _
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields as obj_fields
def remotable(fn):
    """Decorator for remotable object methods."""

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        ctxt = self._context
        if ctxt is None:
            raise exception.OrphanedObjectError(method=fn.__name__, objtype=self.obj_name())
        if self.indirection_api:
            updates, result = self.indirection_api.object_action(ctxt, self, fn.__name__, args, kwargs)
            for key, value in updates.items():
                if key in self.fields:
                    field = self.fields[key]
                    if isinstance(value, VersionedObject):
                        setattr(self, key, value)
                    else:
                        setattr(self, key, field.from_primitive(self, key, value))
            self.obj_reset_changes()
            self._changed_fields = set(updates.get('obj_what_changed', []))
            return result
        else:
            return fn(self, *args, **kwargs)
    wrapper.remotable = True
    wrapper.original_fn = fn
    return wrapper