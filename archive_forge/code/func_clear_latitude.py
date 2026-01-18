from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_latitude(self):
    if self.has_latitude_:
        self.has_latitude_ = 0
        self.latitude_ = 0.0