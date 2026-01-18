from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def mutable_key_value(self):
    self.has_key_value_ = 1
    return self.key_value()