import boto
from boto.compat import json
from boto.cloudsearch.optionstatus import OptionStatus
from boto.cloudsearch.optionstatus import IndexFieldStatus
from boto.cloudsearch.optionstatus import ServicePoliciesStatus
from boto.cloudsearch.optionstatus import RankExpressionStatus
from boto.cloudsearch.document import DocumentServiceConnection
from boto.cloudsearch.search import SearchConnection
def get_index_fields(self, field_names=None):
    """
        Return a list of index fields defined for this domain.
        """
    data = self.layer1.describe_index_fields(self.name, field_names)
    return [IndexFieldStatus(self, d) for d in data]