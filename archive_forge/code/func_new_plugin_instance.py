import pkgutil
import sys
from oslo_concurrency import lockutils
from oslo_log import log as logging
from oslo_utils import importutils
from stevedore import driver
from stevedore import enabled
from neutron_lib._i18n import _
def new_plugin_instance(self, plugin_name, *args, **kwargs):
    """Create a new instance of a plugin.

        :param plugin_name: The name of the plugin to instantiate.
        :param args: Any args to pass onto the constructor.
        :param kwargs: Any kwargs to pass onto the constructor.
        :returns: A new instance of plugin_name.
        :raises: KeyError if plugin_name is not loaded.
        """
    self._assert_plugin_loaded(plugin_name)
    return self.get_plugin_class(plugin_name)(*args, **kwargs)