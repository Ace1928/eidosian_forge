import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.cloudsearch2 import exceptions
def delete_analysis_scheme(self, domain_name, analysis_scheme_name):
    """
        Deletes an analysis scheme. For more information, see
        `Configuring Analysis Schemes`_ in the Amazon CloudSearch
        Developer Guide .

        :type domain_name: string
        :param domain_name: A string that represents the name of a domain.
            Domain names are unique across the domains owned by an account
            within an AWS region. Domain names start with a letter or number
            and can contain the following characters: a-z (lowercase), 0-9, and
            - (hyphen).

        :type analysis_scheme_name: string
        :param analysis_scheme_name: The name of the analysis scheme you want
            to delete.

        """
    params = {'DomainName': domain_name, 'AnalysisSchemeName': analysis_scheme_name}
    return self._make_request(action='DeleteAnalysisScheme', verb='POST', path='/', params=params)