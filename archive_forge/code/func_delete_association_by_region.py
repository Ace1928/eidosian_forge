import abc
from keystone import exception
@abc.abstractmethod
def delete_association_by_region(self, region_id):
    """Remove all the policy associations with the specific region.

        :param region_id: identity of endpoint to check
        :type region_id: string
        :returns: None

        """
    raise exception.NotImplemented()