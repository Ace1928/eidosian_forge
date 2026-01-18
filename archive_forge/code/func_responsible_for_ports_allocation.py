import abc
from neutron_lib.api.definitions import portbindings
def responsible_for_ports_allocation(self, context):
    """Is responsible for a port's resource provider?

        :param context: PortContext instance describing the port
        :returns: True for responsible, False for not responsible

        For ports having an allocation in Placement (as expressed
        in the port's binding:profile.allocation) decide while
        binding if this mechanism driver is responsible for the
        physical network interface represented by the resource
        provider in Placement. Find the resource provider UUID in
        context.current['binding:profile']['allocation'].

        Drivers wanting to support resource allocations for ports in
        Placement (eg. wanting to guarantee some minimum bandwidth)
        must implement this method.

        Default implementation returns False (backward compatibility).
        """
    return False