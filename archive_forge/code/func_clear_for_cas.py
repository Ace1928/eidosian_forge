from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_for_cas(self):
    if self.has_for_cas_:
        self.has_for_cas_ = 0
        self.for_cas_ = 0