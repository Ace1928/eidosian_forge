import abc
from keystone import exception
@abc.abstractmethod
def delete_mapping(self, mapping_id):
    """Delete a mapping.

        :param mapping_id: id of mapping to delete
        :type mapping_ref: string
        :returns: None

        """
    raise exception.NotImplemented()