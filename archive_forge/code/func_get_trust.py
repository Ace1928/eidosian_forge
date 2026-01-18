import abc
from keystone import exception
@abc.abstractmethod
def get_trust(self, trust_id, deleted=False):
    """Get a trust by the trust id.

        :param trust_id: the trust identifier
        :type trust_id: string
        :param deleted: return the trust even if it is deleted, expired, or
                        has no consumptions left
        :type deleted: bool
        """
    raise exception.NotImplemented()