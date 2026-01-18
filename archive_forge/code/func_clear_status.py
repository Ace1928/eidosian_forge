from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_status(self):
    if self.has_status_:
        self.has_status_ = 0
        self.status_ = 4