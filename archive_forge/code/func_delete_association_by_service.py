import abc
from keystone import exception
@abc.abstractmethod
def delete_association_by_service(self, service_id):
    """Remove all the policy associations with the specific service.

        :param service_id: identity of endpoint to check
        :type service_id: string
        :returns: None

        """
    raise exception.NotImplemented()