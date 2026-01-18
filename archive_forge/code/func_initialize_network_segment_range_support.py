import abc
from neutron_lib.api.definitions import portbindings
@abc.abstractmethod
def initialize_network_segment_range_support(self):
    """Perform driver network segment range initialization.

        Called during the initialization of the ``network-segment-range``
        service plugin if enabled, after all drivers have been loaded and the
        database has been initialized. This reloads the `default`
        network segment ranges when Neutron server starts/restarts.
        """