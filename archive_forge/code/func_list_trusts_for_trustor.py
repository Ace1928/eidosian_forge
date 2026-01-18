import abc
from keystone import exception
@abc.abstractmethod
def list_trusts_for_trustor(self, trustor, redelegated_trust_id=None):
    raise exception.NotImplemented()