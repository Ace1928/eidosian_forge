from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def set_boolean_value(self, x):
    self.has_boolean_value_ = 1
    self.boolean_value_ = x