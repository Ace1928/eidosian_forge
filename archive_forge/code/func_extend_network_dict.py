import abc
from neutron_lib.api.definitions import portbindings
def extend_network_dict(self, session, base_model, result):
    """Add extended attributes to network dictionary.

        :param session: database session
        :param base_model: network model data
        :param result: network dictionary to extend

        Called inside transaction context on session to add any
        extended attributes defined by this driver to a network
        dictionary to be used for mechanism driver calls and/or
        returned as the result of a network operation.
        """
    pass