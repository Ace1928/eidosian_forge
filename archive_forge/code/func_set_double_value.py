from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def set_double_value(self, x):
    self.has_double_value_ = 1
    self.double_value_ = x