import abc
from keystone import exception
@abc.abstractmethod
def release_registration(self, domain_id, type=None):
    """Release registration if it is held by the domain specified.

        If the specified domain is registered for this domain then free it,
        if it is not then do nothing - no exception is raised.

        :param domain_id: the domain in question
        :param type: type of registration, if None then all registrations
                     for this domain will be freed

        """
    raise exception.NotImplemented()