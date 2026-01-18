import abc
from keystone import exception
@abc.abstractmethod
def list_federated_users_info(self, hints=None):
    """Get the shadow users info with the specified filters.

        :param hints: contains the list of filters yet to be satisfied.
                      Any filters satisfied here will be removed so that
                      the caller will know if any filters remain.
        :returns list: A list of objects that containing the shadow users
                       reference.

        """
    raise exception.NotImplemented()