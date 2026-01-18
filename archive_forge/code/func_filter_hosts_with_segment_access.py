import abc
from neutron_lib.api.definitions import portbindings
def filter_hosts_with_segment_access(self, context, segments, candidate_hosts, agent_getter):
    """Filter hosts with access to at least one segment.

        :returns: a set with a subset of candidate_hosts.

        A driver can overload this method to return a subset of candidate_hosts
        with the ones with access to at least one segment.

        Default implementation returns all hosts to disable filtering
        (backward compatibility).
        """
    return candidate_hosts