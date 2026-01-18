import abc
from keystone.common import provider_api
from keystone import exception
@abc.abstractmethod
def purge_mappings(self, purge_filter):
    """Purge selected identity mappings.

        :param dict purge_filter: Containing the attributes of the filter that
                                  defines which entries to purge. An empty
                                  filter means purge all mappings.

        """
    raise exception.NotImplemented()