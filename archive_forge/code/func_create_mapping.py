import abc
from keystone import exception
@abc.abstractmethod
def create_mapping(self, mapping_id, mapping):
    """Create a mapping.

        :param mapping_id: ID of mapping object
        :type mapping_id: string
        :param mapping: mapping ref with mapping name
        :type mapping: dict
        :returns: mapping ref
        :rtype: dict

        """
    raise exception.NotImplemented()