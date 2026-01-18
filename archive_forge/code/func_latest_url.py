from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.run import k8s_object
from googlecloudsdk.api_lib.run import revision
from googlecloudsdk.api_lib.run import traffic
@property
def latest_url(self):
    """A url at which we can reach the latest ready revision."""
    for target in self.status.traffic:
        if self._ShouldIncludeInLatestPercent(target) and target.url:
            return target.url
    return None