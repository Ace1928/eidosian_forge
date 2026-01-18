import abc
from neutron_lib.api.definitions import portbindings
def extend_port_dict(self, session, base_model, result):
    """Add extended attributes to port dictionary.

        :param session: database session
        :param base_model: port model data
        :param result: port dictionary to extend

        Called inside transaction context on session to add any
        extended attributes defined by this driver to a port
        dictionary to be used for mechanism driver calls
        and/or returned as the result of a port operation.
        """
    pass