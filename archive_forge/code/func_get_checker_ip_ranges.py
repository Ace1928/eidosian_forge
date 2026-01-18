from boto.route53 import exception
import random
import uuid
import xml.sax
import boto
from boto.connection import AWSAuthConnection
from boto import handler
import boto.jsonresponse
from boto.route53.record import ResourceRecordSets
from boto.route53.zone import Zone
from boto.compat import six, urllib
def get_checker_ip_ranges(self):
    """
        Return a list of Route53 healthcheck IP ranges
        """
    uri = '/%s/checkeripranges' % self.Version
    response = self.make_request('GET', uri)
    body = response.read()
    boto.log.debug(body)
    if response.status >= 300:
        raise exception.DNSServerError(response.status, response.reason, body)
    e = boto.jsonresponse.Element(list_marker='CheckerIpRanges', item_marker=('member',))
    h = boto.jsonresponse.XmlHandler(e, None)
    h.parse(body)
    return e