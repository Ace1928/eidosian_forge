from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_double_value(self):
    if self.has_double_value_:
        self.has_double_value_ = 0
        self.double_value_ = 0.0