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
def create_hosted_zone(self, domain_name, caller_ref=None, comment='', private_zone=False, vpc_id=None, vpc_region=None):
    """
        Create a new Hosted Zone.  Returns a Python data structure with
        information about the newly created Hosted Zone.

        :type domain_name: str
        :param domain_name: The name of the domain. This should be a
            fully-specified domain, and should end with a final period
            as the last label indication.  If you omit the final period,
            Amazon Route 53 assumes the domain is relative to the root.
            This is the name you have registered with your DNS registrar.
            It is also the name you will delegate from your registrar to
            the Amazon Route 53 delegation servers returned in
            response to this request.A list of strings with the image
            IDs wanted.

        :type caller_ref: str
        :param caller_ref: A unique string that identifies the request
            and that allows failed CreateHostedZone requests to be retried
            without the risk of executing the operation twice.  If you don't
            provide a value for this, boto will generate a Type 4 UUID and
            use that.

        :type comment: str
        :param comment: Any comments you want to include about the hosted
            zone.

        :type private_zone: bool
        :param private_zone: Set True if creating a private hosted zone.

        :type vpc_id: str
        :param vpc_id: When creating a private hosted zone, the VPC Id to
            associate to is required.

        :type vpc_region: str
        :param vpc_region: When creating a private hosted zone, the region
            of the associated VPC is required.

        """
    if caller_ref is None:
        caller_ref = str(uuid.uuid4())
    if private_zone:
        params = {'name': domain_name, 'caller_ref': caller_ref, 'comment': comment, 'vpc_id': vpc_id, 'vpc_region': vpc_region, 'xmlns': self.XMLNameSpace}
        xml_body = HZPXML % params
    else:
        params = {'name': domain_name, 'caller_ref': caller_ref, 'comment': comment, 'xmlns': self.XMLNameSpace}
        xml_body = HZXML % params
    uri = '/%s/hostedzone' % self.Version
    response = self.make_request('POST', uri, {'Content-Type': 'text/xml'}, xml_body)
    body = response.read()
    boto.log.debug(body)
    if response.status == 201:
        e = boto.jsonresponse.Element(list_marker='NameServers', item_marker=('NameServer',))
        h = boto.jsonresponse.XmlHandler(e, None)
        h.parse(body)
        return e
    else:
        raise exception.DNSServerError(response.status, response.reason, body)