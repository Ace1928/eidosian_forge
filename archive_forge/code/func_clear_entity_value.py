from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_entity_value(self):
    if self.has_entity_value_:
        self.has_entity_value_ = 0
        if self.entity_value_ is not None:
            self.entity_value_.Clear()