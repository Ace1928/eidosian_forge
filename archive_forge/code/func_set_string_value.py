from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def set_string_value(self, x):
    self.has_string_value_ = 1
    self.string_value_ = x