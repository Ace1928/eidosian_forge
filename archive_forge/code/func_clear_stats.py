from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_stats(self):
    if self.has_stats_:
        self.has_stats_ = 0
        if self.stats_ is not None:
            self.stats_.Clear()