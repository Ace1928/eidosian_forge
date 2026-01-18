import abc
from neutron_lib.api.definitions import portbindings
@abc.abstractmethod
def set_binding(self, segment_id, vif_type, vif_details, status=None):
    """Set the bottom-level binding for the port.

        :param segment_id: Network segment bound for the port.
        :param vif_type: The VIF type for the bound port.
        :param vif_details: Dictionary with details for VIF driver.
        :param status: Port status to set if not None.

        This method is called by MechanismDriver.bind_port to indicate
        success and specify binding details to use for port. The
        segment_id must identify an item in the current value of the
        segments_to_bind property.
        """