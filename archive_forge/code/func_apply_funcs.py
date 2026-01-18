import collections
import inspect
from oslo_log import log as logging
from oslo_utils import timeutils
from neutron_lib.utils import helpers
@timeutils.time_it(LOG, min_duration=0.1)
def apply_funcs(resource_type, response, db_object):
    """Appy registered functions for the said resource type.

    :param resource_type: The resource type to apply funcs for.
    :param response: The response object.
    :param db_object: The Database object.
    :returns: None
    """
    for func in get_funcs(resource_type):
        resolved_func = helpers.resolve_ref(func)
        if resolved_func:
            resolved_func(response, db_object)