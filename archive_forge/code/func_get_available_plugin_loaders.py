import abc
import stevedore
from keystoneauth1 import exceptions
def get_available_plugin_loaders():
    """Retrieve all the plugin classes available on the system.

    :returns: A dict with plugin entrypoint name as the key and the plugin
              loader as the value.
    :rtype: dict
    """
    mgr = stevedore.EnabledExtensionManager(namespace=PLUGIN_NAMESPACE, check_func=_auth_plugin_available, invoke_on_load=True, propagate_map_exceptions=True)
    return dict(mgr.map(lambda ext: (ext.entry_point.name, ext.obj)))