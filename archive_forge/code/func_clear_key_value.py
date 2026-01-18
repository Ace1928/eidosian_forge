from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_key_value(self):
    if self.has_key_value_:
        self.has_key_value_ = 0
        if self.key_value_ is not None:
            self.key_value_.Clear()