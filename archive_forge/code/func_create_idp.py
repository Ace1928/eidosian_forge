import abc
from keystone import exception
@abc.abstractmethod
def create_idp(self, idp_id, idp):
    """Create an identity provider.

        :param idp_id: ID of IdP object
        :type idp_id: string
        :param idp: idp object
        :type idp: dict
        :returns: idp ref
        :rtype: dict

        """
    raise exception.NotImplemented()