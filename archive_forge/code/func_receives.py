import collections
import inspect
from neutron_lib._i18n import _
from neutron_lib.callbacks import manager
from neutron_lib.callbacks import priority_group
def receives(resource, events, priority=priority_group.PRIORITY_DEFAULT):
    """Use to decorate methods on classes before initialization.

    Any classes that use this must themselves be decorated with the
    @has_registry_receivers decorator to setup the __new__ method to
    actually register the instance methods after initialization.
    """
    if not isinstance(events, (list, tuple, set)):
        msg = _("'events' must be a collection (list, tuple, set)")
        raise AssertionError(msg)

    def decorator(f):
        for e in events:
            _REGISTERED_CLASS_METHODS[f].append((resource, e, priority))
        return f
    return decorator