from boto.cloudsearch2.optionstatus import IndexFieldStatus
from boto.cloudsearch2.optionstatus import ServicePoliciesStatus
from boto.cloudsearch2.optionstatus import ExpressionStatus
from boto.cloudsearch2.optionstatus import AvailabilityOptionsStatus
from boto.cloudsearch2.optionstatus import ScalingParametersStatus
from boto.cloudsearch2.document import DocumentServiceConnection
from boto.cloudsearch2.search import SearchConnection
def get_analysis_schemes(self):
    """
        Return a list of Analysis Scheme objects.
        """
    return self.layer1.describe_analysis_schemes(self.name)