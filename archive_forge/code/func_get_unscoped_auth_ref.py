import abc
from keystoneauth1.identity.v3 import base
from keystoneauth1.identity.v3 import token
@abc.abstractmethod
def get_unscoped_auth_ref(self, session, **kwargs):
    """Fetch unscoped federated token."""