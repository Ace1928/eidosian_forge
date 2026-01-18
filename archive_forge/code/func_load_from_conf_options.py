from keystoneauth1.loading import base
from keystoneauth1.loading import opts
def load_from_conf_options(conf, group, **kwargs):
    """Load a plugin from an oslo_config CONF object.

    Each plugin will register their own required options and so there is no
    standard list and the plugin should be consulted.

    The base options should have been registered with register_conf_options
    before this function is called.

    :param conf: A conf object.
    :type conf: oslo_config.cfg.ConfigOpts
    :param str group: The group name that options should be read from.

    :returns: An authentication Plugin or None if a name is not provided
    :rtype: :class:`keystoneauth1.plugin.BaseAuthPlugin`

    :raises keystoneauth1.exceptions.auth_plugins.NoMatchingPlugin:
        if a plugin cannot be created.
    """
    if conf[group].auth_section:
        group = conf[group].auth_section
    name = conf[group].auth_type
    if not name:
        return None
    plugin = base.get_plugin_loader(name)
    plugin_opts = plugin.get_options()
    oslo_opts = [o._to_oslo_opt() for o in plugin_opts]
    conf.register_opts(oslo_opts, group=group)

    def _getter(opt):
        return conf[group][opt.dest]
    return plugin.load_from_options_getter(_getter, **kwargs)