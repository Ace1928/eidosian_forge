import abc
from neutron_lib.api.definitions import portbindings
@property
def extension_alias(self):
    """Supported extension alias.

        Return the alias identifying the core API extension supported
        by this driver. Do not declare if API extension handling will
        be left to a service plugin, and we just need to provide
        core resource extension and updates.
        """
    return