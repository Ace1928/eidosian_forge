import abc
from keystone.common import provider_api
from keystone import exception
@abc.abstractmethod
def get_domain_mapping_list(self, domain_id, entity_type=None):
    """Return mappings for the domain.

        :param domain_id: Domain ID to get mappings for.
        :param entity_type: Optional entity_type to get mappings for.
        :type entity_type: String, one of mappings defined in
            keystone.identity.mapping_backends.mapping.EntityType
        :returns: list of mappings.
        """
    raise exception.NotImplemented()