import boto
import boto.jsonresponse
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
def delete_rank_expression(self, domain_name, rank_name):
    """
        Deletes an existing ``RankExpression`` from the search domain.

        :type domain_name: string
        :param domain_name: A string that represents the name of a
            domain. Domain names must be unique across the domains
            owned by an account within an AWS region. Domain names
            must start with a letter or number and can contain the
            following characters: a-z (lowercase), 0-9, and -
            (hyphen). Uppercase letters and underscores are not
            allowed.

        :type rank_name: string
        :param rank_name: Name of the ``RankExpression`` to delete.

        :raises: BaseException, InternalException, ResourceNotFoundException
        """
    doc_path = ('delete_rank_expression_response', 'delete_rank_expression_result', 'rank_expression')
    params = {'DomainName': domain_name, 'RankName': rank_name}
    return self.get_response(doc_path, 'DeleteRankExpression', params, verb='POST')