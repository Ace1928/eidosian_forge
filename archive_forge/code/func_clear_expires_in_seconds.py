from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_expires_in_seconds(self):
    if self.has_expires_in_seconds_:
        self.has_expires_in_seconds_ = 0
        self.expires_in_seconds_ = 0