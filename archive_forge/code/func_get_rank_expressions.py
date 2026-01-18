import boto
from boto.compat import json
from boto.cloudsearch.optionstatus import OptionStatus
from boto.cloudsearch.optionstatus import IndexFieldStatus
from boto.cloudsearch.optionstatus import ServicePoliciesStatus
from boto.cloudsearch.optionstatus import RankExpressionStatus
from boto.cloudsearch.document import DocumentServiceConnection
from boto.cloudsearch.search import SearchConnection
def get_rank_expressions(self, rank_names=None):
    """
        Return a list of rank expressions defined for this domain.
        """
    fn = self.layer1.describe_rank_expressions
    data = fn(self.name, rank_names)
    return [RankExpressionStatus(self, d, fn) for d in data]