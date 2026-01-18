import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.cloudsearch2 import exceptions
def describe_analysis_schemes(self, domain_name, analysis_scheme_names=None, deployed=None):
    """
        Gets the analysis schemes configured for a domain. An analysis
        scheme defines language-specific text processing options for a
        `text` field. Can be limited to specific analysis schemes by
        name. By default, shows all analysis schemes and includes any
        pending changes to the configuration. Set the `Deployed`
        option to `True` to show the active configuration and exclude
        pending changes. For more information, see `Configuring
        Analysis Schemes`_ in the Amazon CloudSearch Developer Guide .

        :type domain_name: string
        :param domain_name: The name of the domain you want to describe.

        :type analysis_scheme_names: list
        :param analysis_scheme_names: The analysis schemes you want to
            describe.

        :type deployed: boolean
        :param deployed: Whether to display the deployed configuration (
            `True`) or include any pending changes ( `False`). Defaults to
            `False`.

        """
    params = {'DomainName': domain_name}
    if analysis_scheme_names is not None:
        self.build_list_params(params, analysis_scheme_names, 'AnalysisSchemeNames.member')
    if deployed is not None:
        params['Deployed'] = str(deployed).lower()
    return self._make_request(action='DescribeAnalysisSchemes', verb='POST', path='/', params=params)