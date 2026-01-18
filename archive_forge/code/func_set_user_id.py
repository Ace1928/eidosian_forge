from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def set_user_id(self, x):
    self.has_user_id_ = 1
    self.user_id_ = x