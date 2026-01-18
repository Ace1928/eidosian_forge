import xml.sax
import threading
import boto
from boto import handler
from boto.connection import AWSQueryConnection
from boto.sdb.domain import Domain, DomainMetaData
from boto.sdb.item import Item
from boto.sdb.regioninfo import SDBRegionInfo
from boto.exception import SDBResponseError
def get_domain_and_name(self, domain_or_name):
    """
        Given a ``str`` or :class:`boto.sdb.domain.Domain`, return a
        ``tuple`` with the following members (in order):

            * In instance of :class:`boto.sdb.domain.Domain` for the requested
              domain
            * The domain's name as a ``str``

        :type domain_or_name: ``str`` or :class:`boto.sdb.domain.Domain`
        :param domain_or_name: The domain or domain name to get the domain
            and name for.

        :raises: :class:`boto.exception.SDBResponseError` when an invalid
            domain name is specified.

        :rtype: tuple
        :return: A ``tuple`` with contents outlined as per above.
        """
    if isinstance(domain_or_name, Domain):
        return (domain_or_name, domain_or_name.name)
    else:
        return (self.get_domain(domain_or_name), domain_or_name)