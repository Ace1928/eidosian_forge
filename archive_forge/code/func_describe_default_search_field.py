import boto
import boto.jsonresponse
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
def describe_default_search_field(self, domain_name):
    """
        Describes options defining the default search field used by
        indexing for the search domain.

        :type domain_name: string
        :param domain_name: A string that represents the name of a
            domain. Domain names must be unique across the domains
            owned by an account within an AWS region. Domain names
            must start with a letter or number and can contain the
            following characters: a-z (lowercase), 0-9, and -
            (hyphen). Uppercase letters and underscores are not
            allowed.

        :raises: BaseException, InternalException, ResourceNotFoundException
        """
    doc_path = ('describe_default_search_field_response', 'describe_default_search_field_result', 'default_search_field')
    params = {'DomainName': domain_name}
    return self.get_response(doc_path, 'DescribeDefaultSearchField', params, verb='POST')