import abc
from keystone import exception
@abc.abstractmethod
def list_trusts_for_trustee(self, trustee):
    raise exception.NotImplemented()