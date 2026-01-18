import abc
import argparse
import os
from stevedore import extension
from troveclient.apiclient import exceptions
def discover_auth_systems():
    """Discover the available auth-systems.

    This won't take into account the old style auth-systems.
    """
    global _discovered_plugins
    _discovered_plugins = {}

    def add_plugin(ext):
        _discovered_plugins[ext.name] = ext.plugin
    ep_namespace = 'troveclient.apiclient.auth'
    mgr = extension.ExtensionManager(ep_namespace)
    mgr.map(add_plugin)