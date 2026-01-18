from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_qps(self):
    if self.has_qps_:
        self.has_qps_ = 0
        self.qps_ = 0.0