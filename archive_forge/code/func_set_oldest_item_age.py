from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def set_oldest_item_age(self, x):
    self.has_oldest_item_age_ = 1
    self.oldest_item_age_ = x