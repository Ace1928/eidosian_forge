from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def mutable_stats(self):
    self.has_stats_ = 1
    return self.stats()