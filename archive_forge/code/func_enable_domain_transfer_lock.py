import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.route53.domains import exceptions
def enable_domain_transfer_lock(self, domain_name):
    """
        This operation sets the transfer lock on the domain
        (specifically the `clientTransferProhibited` status) to
        prevent domain transfers. Successful submission returns an
        operation ID that you can use to track the progress and
        completion of the action. If the request is not completed
        successfully, the domain registrant will be notified by email.

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
    return self.make_request(action='EnableDomainTransferLock', body=json.dumps(params))