import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.route53.domains import exceptions
def retrieve_domain_auth_code(self, domain_name):
    """
        This operation returns the AuthCode for the domain. To
        transfer a domain to another registrar, you provide this value
        to the new registrar.

        :type domain_name: string
        :param domain_name: The name of a domain.
        Type: String

        Default: None

        Constraints: The domain name can contain only the letters a through z,
            the numbers 0 through 9, and hyphen (-). Internationalized Domain
            Names are not supported.

        Required: Yes

        """
    params = {'DomainName': domain_name}
    return self.make_request(action='RetrieveDomainAuthCode', body=json.dumps(params))