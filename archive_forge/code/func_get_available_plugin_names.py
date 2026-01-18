import abc
import stevedore
from keystoneauth1 import exceptions
def get_available_plugin_names():
    """Get the names of all the plugins that are available on the system.

    This is particularly useful for help and error text to prompt a user for
    example what plugins they may specify.

    :returns: A list of names.
    :rtype: frozenset
    """
    mgr = stevedore.EnabledExtensionManager(namespace=PLUGIN_NAMESPACE, check_func=_auth_plugin_available, invoke_on_load=True, propagate_map_exceptions=True)
    return frozenset(mgr.names())