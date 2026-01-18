import abc
from neutron_lib.api.definitions import portbindings
@property
def connectivity(self):
    """Return the mechanism driver connectivity type

        The possible values are "l2", "l3" and "legacy" (default).

        :returns: a string in ("l2", "l3", "legacy")
        """
    return portbindings.CONNECTIVITY_LEGACY