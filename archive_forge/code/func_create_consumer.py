import abc
import string
from keystone import exception
@abc.abstractmethod
def create_consumer(self, consumer_ref):
    """Create consumer.

        :param consumer_ref: consumer ref with consumer name
        :type consumer_ref: dict
        :returns: consumer_ref

        """
    raise exception.NotImplemented()