from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_initial_flags(self):
    if self.has_initial_flags_:
        self.has_initial_flags_ = 0
        self.initial_flags_ = 0