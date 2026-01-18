import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.cloudsearch2 import exceptions
def describe_expressions(self, domain_name, expression_names=None, deployed=None):
    """
        Gets the expressions configured for the search domain. Can be
        limited to specific expressions by name. By default, shows all
        expressions and includes any pending changes to the
        configuration. Set the `Deployed` option to `True` to show the
        active configuration and exclude pending changes. For more
        information, see `Configuring Expressions`_ in the Amazon
        CloudSearch Developer Guide .

        :type domain_name: string
        :param domain_name: The name of the domain you want to describe.

        :type expression_names: list
        :param expression_names: Limits the `DescribeExpressions` response to
            the specified expressions. If not specified, all expressions are
            shown.

        :type deployed: boolean
        :param deployed: Whether to display the deployed configuration (
            `True`) or include any pending changes ( `False`). Defaults to
            `False`.

        """
    params = {'DomainName': domain_name}
    if expression_names is not None:
        self.build_list_params(params, expression_names, 'ExpressionNames.member')
    if deployed is not None:
        params['Deployed'] = str(deployed).lower()
    return self._make_request(action='DescribeExpressions', verb='POST', path='/', params=params)