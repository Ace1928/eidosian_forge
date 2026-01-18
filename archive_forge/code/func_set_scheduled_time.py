from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def set_scheduled_time(self, x):
    self.has_scheduled_time_ = 1
    self.scheduled_time_ = x