from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_increment_status(self):
    if self.has_increment_status_:
        self.has_increment_status_ = 0
        self.increment_status_ = 0