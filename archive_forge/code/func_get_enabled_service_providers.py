import abc
from keystone import exception
@abc.abstractmethod
def get_enabled_service_providers(self):
    """List enabled service providers for Service Catalog.

        Service Provider in a catalog contains three attributes: ``id``,
        ``auth_url``, ``sp_url``, where:

        - id is a unique, user defined identifier for service provider object
        - auth_url is an authentication URL of remote Keystone
        - sp_url a URL accessible at the remote service provider where SAML
          assertion is transmitted.

        :returns: list of dictionaries with enabled service providers
        :rtype: list of dicts

        """
    raise exception.NotImplemented()