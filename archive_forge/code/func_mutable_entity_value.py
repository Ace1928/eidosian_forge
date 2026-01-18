from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def mutable_entity_value(self):
    self.has_entity_value_ = 1
    return self.entity_value()