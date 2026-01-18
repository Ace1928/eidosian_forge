import xml.sax
import threading
import boto
from boto import handler
from boto.connection import AWSQueryConnection
from boto.sdb.domain import Domain, DomainMetaData
from boto.sdb.item import Item
from boto.sdb.regioninfo import SDBRegionInfo
from boto.exception import SDBResponseError
def domain_metadata(self, domain_or_name):
    """
        Get the Metadata for a SimpleDB domain.

        :type domain_or_name: string or :class:`boto.sdb.domain.Domain` object.
        :param domain_or_name: Either the name of a domain or a Domain object

        :rtype: :class:`boto.sdb.domain.DomainMetaData` object
        :return: The newly created domain metadata object
        """
    domain, domain_name = self.get_domain_and_name(domain_or_name)
    params = {'DomainName': domain_name}
    d = self.get_object('DomainMetadata', params, DomainMetaData)
    d.domain = domain
    return d