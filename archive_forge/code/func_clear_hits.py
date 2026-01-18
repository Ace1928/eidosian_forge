from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_hits(self):
    if self.has_hits_:
        self.has_hits_ = 0
        self.hits_ = 0