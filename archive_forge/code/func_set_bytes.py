from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def set_bytes(self, x):
    self.has_bytes_ = 1
    self.bytes_ = x