from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def mutable_override(self):
    self.has_override_ = 1
    return self.override()