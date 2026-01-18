from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def set_new_value(self, x):
    self.has_new_value_ = 1
    self.new_value_ = x