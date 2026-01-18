import boto
import boto.jsonresponse
from boto.compat import json
from boto.regioninfo import RegionInfo
from boto.connection import AWSQueryConnection
def check_dns_availability(self, cname_prefix):
    """Checks if the specified CNAME is available.

        :type cname_prefix: string
        :param cname_prefix: The prefix used when this CNAME is
            reserved.
        """
    params = {'CNAMEPrefix': cname_prefix}
    return self._get_response('CheckDNSAvailability', params)