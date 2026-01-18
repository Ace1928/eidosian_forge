import abc
from neutron_lib._i18n import _
from neutron_lib import constants
def get_plugin_interface(self):
    """Returns an abstract class which defines contract for the plugin.

        The abstract class should inherit from
        neutron_lib.services.base.ServicePluginBase.
        Methods in this abstract class should be decorated as abstractmethod
        """