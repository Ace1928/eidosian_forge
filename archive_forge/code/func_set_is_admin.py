from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def set_is_admin(self, x):
    self.has_is_admin_ = 1
    self.is_admin_ = x