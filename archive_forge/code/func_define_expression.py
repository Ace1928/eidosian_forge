import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.cloudsearch2 import exceptions
def define_expression(self, domain_name, expression):
    """
        Configures an `Expression` for the search domain. Used to
        create new expressions and modify existing ones. If the
        expression exists, the new configuration replaces the old one.
        For more information, see `Configuring Expressions`_ in the
        Amazon CloudSearch Developer Guide .

        :type domain_name: string
        :param domain_name: A string that represents the name of a domain.
            Domain names are unique across the domains owned by an account
            within an AWS region. Domain names start with a letter or number
            and can contain the following characters: a-z (lowercase), 0-9, and
            - (hyphen).

        :type expression: dict
        :param expression: A named expression that can be evaluated at search
            time. Can be used to sort the search results, define other
            expressions, or return computed information in the search results.

        """
    params = {'DomainName': domain_name}
    self.build_complex_param(params, 'Expression', expression)
    return self._make_request(action='DefineExpression', verb='POST', path='/', params=params)