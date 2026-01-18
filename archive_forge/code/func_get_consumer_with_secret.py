import abc
import string
from keystone import exception
@abc.abstractmethod
def get_consumer_with_secret(self, consumer_id):
    """Like get_consumer(), but also returns consumer secret.

        Returned dictionary consumer_ref includes consumer secret.
        Secrets should only be shared upon consumer creation; the
        consumer secret is required to verify incoming OAuth requests.

        :param consumer_id: id of consumer to get
        :type consumer_id: string
        :returns: consumer_ref containing consumer secret

        """
    raise exception.NotImplemented()