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
def delete_hosted_zone(self, hosted_zone_id):
    """
        Delete the hosted zone specified by the given id.

        :type hosted_zone_id: str
        :param hosted_zone_id: The hosted zone's id

        """
    uri = '/%s/hostedzone/%s' % (self.Version, hosted_zone_id)
    response = self.make_request('DELETE', uri)
    body = response.read()
    boto.log.debug(body)
    if response.status not in (200, 204):
        raise exception.DNSServerError(response.status, response.reason, body)
    e = boto.jsonresponse.Element()
    h = boto.jsonresponse.XmlHandler(e, None)
    h.parse(body)
    return e