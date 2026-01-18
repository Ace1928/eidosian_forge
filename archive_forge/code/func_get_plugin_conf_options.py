from keystoneauth1.loading import base
from keystoneauth1.loading import opts
def get_plugin_conf_options(plugin):
    """Get the oslo_config options for a specific plugin.

    This will be the list of config options that is registered and loaded by
    the specified plugin.

    :param plugin: The name of the plugin loader or a plugin loader object
    :type plugin: str or keystoneauth1._loading.BaseLoader

    :returns: A list of oslo_config options.
    """
    try:
        getter = plugin.get_options
    except AttributeError:
        opts = base.get_plugin_options(plugin)
    else:
        opts = getter()
    return [o._to_oslo_opt() for o in opts]