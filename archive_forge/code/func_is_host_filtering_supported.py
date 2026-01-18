import abc
from neutron_lib.api.definitions import portbindings
@classmethod
def is_host_filtering_supported(cls):
    return cls.filter_hosts_with_segment_access != MechanismDriver.filter_hosts_with_segment_access