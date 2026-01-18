import boto
from boto.compat import json
from boto.cloudsearch.optionstatus import OptionStatus
from boto.cloudsearch.optionstatus import IndexFieldStatus
from boto.cloudsearch.optionstatus import ServicePoliciesStatus
from boto.cloudsearch.optionstatus import RankExpressionStatus
from boto.cloudsearch.document import DocumentServiceConnection
from boto.cloudsearch.search import SearchConnection
def get_access_policies(self):
    """
        Return a :class:`boto.cloudsearch.option.OptionStatus` object
        representing the currently defined access policies for
        the domain.
        """
    return ServicePoliciesStatus(self, None, self.layer1.describe_service_access_policies, self.layer1.update_service_access_policies)