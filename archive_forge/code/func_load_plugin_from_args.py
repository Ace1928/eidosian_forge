import abc
import argparse
import os
from stevedore import extension
from troveclient.apiclient import exceptions
def load_plugin_from_args(args):
    """Load required plugin and populate it with options.

    Try to guess auth system if it is not specified. Systems are tried in
    alphabetical order.

    :type args: argparse.Namespace
    :raises: AuthPluginOptionsMissing
    """
    auth_system = args.os_auth_system
    if auth_system:
        plugin = load_plugin(auth_system)
        plugin.parse_opts(args)
        plugin.sufficient_options()
        return plugin
    for plugin_auth_system in sorted(_discovered_plugins.keys()):
        plugin_class = _discovered_plugins[plugin_auth_system]
        plugin = plugin_class()
        plugin.parse_opts(args)
        try:
            plugin.sufficient_options()
        except exceptions.AuthPluginOptionsMissing:
            continue
        return plugin
    raise exceptions.AuthPluginOptionsMissing(['auth_system'])