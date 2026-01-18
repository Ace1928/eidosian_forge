from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.run import k8s_object
from googlecloudsdk.api_lib.run import revision
from googlecloudsdk.api_lib.run import traffic
@property
def status_traffic(self):
    self.AssertFullObject()
    return traffic.TrafficTargets(self._messages, [] if self.status is None else self.status.traffic)