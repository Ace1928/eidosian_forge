import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.cloudsearch2 import exceptions
def delete_suggester(self, domain_name, suggester_name):
    """
        Deletes a suggester. For more information, see `Getting Search
        Suggestions`_ in the Amazon CloudSearch Developer Guide .

        :type domain_name: string
        :param domain_name: A string that represents the name of a domain.
            Domain names are unique across the domains owned by an account
            within an AWS region. Domain names start with a letter or number
            and can contain the following characters: a-z (lowercase), 0-9, and
            - (hyphen).

        :type suggester_name: string
        :param suggester_name: Specifies the name of the suggester you want to
            delete.

        """
    params = {'DomainName': domain_name, 'SuggesterName': suggester_name}
    return self._make_request(action='DeleteSuggester', verb='POST', path='/', params=params)