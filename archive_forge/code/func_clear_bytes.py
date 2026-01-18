from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_bytes(self):
    if self.has_bytes_:
        self.has_bytes_ = 0
        self.bytes_ = 0