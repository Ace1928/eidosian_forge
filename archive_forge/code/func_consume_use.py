import abc
from keystone import exception
@abc.abstractmethod
def consume_use(self, trust_id):
    """Consume one use of a trust.

        One use of a trust is consumed when the trust was created with a
        limitation on its uses, provided there are still uses available.

        :raises keystone.exception.TrustUseLimitReached: If no remaining uses
            for trust.
        :raises keystone.exception.TrustNotFound: If the trust doesn't exist.
        """
    raise exception.NotImplemented()