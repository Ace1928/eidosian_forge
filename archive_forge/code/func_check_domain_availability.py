import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.route53.domains import exceptions
def check_domain_availability(self, domain_name, idn_lang_code=None):
    """
        This operation checks the availability of one domain name. You
        can access this API without authenticating. Note that if the
        availability status of a domain is pending, you must submit
        another request to determine the availability of the domain
        name.

        :type domain_name: string
        :param domain_name: The name of a domain.
        Type: String

        Default: None

        Constraints: The domain name can contain only the letters a through z,
            the numbers 0 through 9, and hyphen (-). Internationalized Domain
            Names are not supported.

        Required: Yes

        :type idn_lang_code: string
        :param idn_lang_code: Reserved for future use.

        """
    params = {'DomainName': domain_name}
    if idn_lang_code is not None:
        params['IdnLangCode'] = idn_lang_code
    return self.make_request(action='CheckDomainAvailability', body=json.dumps(params))