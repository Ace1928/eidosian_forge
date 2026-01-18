from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def set_indexed(self, x):
    self.has_indexed_ = 1
    self.indexed_ = x