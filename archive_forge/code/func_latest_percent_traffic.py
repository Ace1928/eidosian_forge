from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.run import k8s_object
from googlecloudsdk.api_lib.run import revision
from googlecloudsdk.api_lib.run import traffic
@property
def latest_percent_traffic(self):
    """The percent of traffic the latest ready revision is serving."""
    return sum((target.percent or 0 for target in self.status.traffic if self._ShouldIncludeInLatestPercent(target)))