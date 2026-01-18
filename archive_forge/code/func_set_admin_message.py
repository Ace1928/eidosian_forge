from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def set_admin_message(self, x):
    self.has_admin_message_ = 1
    self.admin_message_ = x