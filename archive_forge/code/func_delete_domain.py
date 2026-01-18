import boto
import boto.jsonresponse
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
def delete_domain(self, domain_name):
    """
        Delete a search domain.

        :type domain_name: string
        :param domain_name: A string that represents the name of a
            domain. Domain names must be unique across the domains
            owned by an account within an AWS region. Domain names
            must start with a letter or number and can contain the
            following characters: a-z (lowercase), 0-9, and -
            (hyphen). Uppercase letters and underscores are not
            allowed.

        :raises: BaseException, InternalException
        """
    doc_path = ('delete_domain_response', 'delete_domain_result', 'domain_status')
    params = {'DomainName': domain_name}
    return self.get_response(doc_path, 'DeleteDomain', params, verb='POST')