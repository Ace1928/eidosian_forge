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
def get_hosted_zone_by_name(self, hosted_zone_name):
    """
        Get detailed information about a particular Hosted Zone.

        :type hosted_zone_name: str
        :param hosted_zone_name: The fully qualified domain name for the Hosted
            Zone

        """
    if hosted_zone_name[-1] != '.':
        hosted_zone_name += '.'
    all_hosted_zones = self.get_all_hosted_zones()
    for zone in all_hosted_zones['ListHostedZonesResponse']['HostedZones']:
        if zone['Name'] == hosted_zone_name:
            return self.get_hosted_zone(zone['Id'].split('/')[-1])