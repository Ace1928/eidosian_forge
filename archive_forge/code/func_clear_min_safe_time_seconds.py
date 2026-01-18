from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.entity_v4_pb import *
import googlecloudsdk.third_party.appengine.datastore.entity_v4_pb
def clear_min_safe_time_seconds(self):
    if self.has_min_safe_time_seconds_:
        self.has_min_safe_time_seconds_ = 0
        self.min_safe_time_seconds_ = 0