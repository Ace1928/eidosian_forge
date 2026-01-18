from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_oldest_item_age(self):
    if self.has_oldest_item_age_:
        self.has_oldest_item_age_ = 0
        self.oldest_item_age_ = 0