import abc
import stevedore
from keystoneauth1 import exceptions
def get_plugin_loader(name):
    """Retrieve a plugin class by its entrypoint name.

    :param str name: The name of the object to get.

    :returns: An auth plugin class.
    :rtype: :py:class:`keystoneauth1.loading.BaseLoader`

    :raises keystoneauth1.exceptions.auth_plugins.NoMatchingPlugin:
        if a plugin cannot be created.
    """
    try:
        mgr = stevedore.DriverManager(namespace=PLUGIN_NAMESPACE, invoke_on_load=True, name=name)
    except RuntimeError:
        raise exceptions.NoMatchingPlugin(name)
    return mgr.driver