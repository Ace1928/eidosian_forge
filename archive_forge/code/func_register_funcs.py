import collections
import inspect
from oslo_log import log as logging
from oslo_utils import timeutils
from neutron_lib.utils import helpers
def register_funcs(resource, funcs):
    """Add functions to extend a resource.

    :param resource: A resource collection name.
    :type resource: str

    :param funcs: A list of functions.
    :type funcs: list of callable

    These functions take a resource dict and a resource object and
    update the resource dict with extension data (possibly retrieved
    from the resource db object).
        def _extend_foo_with_bar(foo_res, foo_db):
            foo_res['bar'] = foo_db.bar_info  # example
            return foo_res

    """
    funcs = [helpers.make_weak_ref(f) if callable(f) else f for f in funcs]
    _resource_extend_functions.setdefault(resource, []).extend(funcs)