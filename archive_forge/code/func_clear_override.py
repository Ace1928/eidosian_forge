from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_override(self):
    if self.has_override_:
        self.has_override_ = 0
        if self.override_ is not None:
            self.override_.Clear()