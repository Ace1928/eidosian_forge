import abc
from neutron_lib._i18n import _
from neutron_lib import constants
def is_extension_supported(plugin, alias):
    """Validate that the extension is supported.

    :param plugin: The plugin class.
    :param alias: The alias to check.
    :returns: True if the alias is supported else False.
    """
    return alias in getattr(plugin, 'supported_extension_aliases', [])