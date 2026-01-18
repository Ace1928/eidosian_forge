import abc
from neutron_lib.api.definitions import portbindings
def extend_subnet_dict(self, session, base_model, result):
    """Add extended attributes to subnet dictionary.

        :param session: database session
        :param base_model: subnet model data
        :param result: subnet dictionary to extend

        Called inside transaction context on session to add any
        extended attributes defined by this driver to a subnet
        dictionary to be used for mechanism driver calls and/or
        returned as the result of a subnet operation.
        """
    pass