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
def change_rrsets(self, hosted_zone_id, xml_body):
    """
        Create or change the authoritative DNS information for this
        Hosted Zone.
        Returns a Python data structure with information about the set of
        changes, including the Change ID.

        :type hosted_zone_id: str
        :param hosted_zone_id: The unique identifier for the Hosted Zone

        :type xml_body: str
        :param xml_body: The list of changes to be made, defined in the
            XML schema defined by the Route53 service.

        """
    uri = '/%s/hostedzone/%s/rrset' % (self.Version, hosted_zone_id)
    response = self.make_request('POST', uri, {'Content-Type': 'text/xml'}, xml_body)
    body = response.read()
    boto.log.debug(body)
    if response.status >= 300:
        raise exception.DNSServerError(response.status, response.reason, body)
    e = boto.jsonresponse.Element()
    h = boto.jsonresponse.XmlHandler(e, None)
    h.parse(body)
    return e