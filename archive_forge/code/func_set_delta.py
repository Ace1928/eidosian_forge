from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def set_delta(self, x):
    self.has_delta_ = 1
    self.delta_ = x