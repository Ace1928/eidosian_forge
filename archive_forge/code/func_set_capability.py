from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def set_capability(self, x):
    self.has_capability_ = 1
    self.capability_ = x