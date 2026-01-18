from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_is_admin(self):
    if self.has_is_admin_:
        self.has_is_admin_ = 0
        self.is_admin_ = 0