from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_expiration_time(self):
    if self.has_expiration_time_:
        self.has_expiration_time_ = 0
        self.expiration_time_ = 0