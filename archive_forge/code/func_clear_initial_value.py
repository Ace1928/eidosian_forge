from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_initial_value(self):
    if self.has_initial_value_:
        self.has_initial_value_ = 0
        self.initial_value_ = 0