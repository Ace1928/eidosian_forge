from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_delete_time(self):
    if self.has_delete_time_:
        self.has_delete_time_ = 0
        self.delete_time_ = 0