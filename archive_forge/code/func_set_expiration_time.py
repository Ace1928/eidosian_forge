from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def set_expiration_time(self, x):
    self.has_expiration_time_ = 1
    self.expiration_time_ = x