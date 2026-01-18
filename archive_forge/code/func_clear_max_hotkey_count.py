from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_max_hotkey_count(self):
    if self.has_max_hotkey_count_:
        self.has_max_hotkey_count_ = 0
        self.max_hotkey_count_ = 0