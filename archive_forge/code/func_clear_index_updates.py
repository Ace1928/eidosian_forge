from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.entity_v4_pb import *
import googlecloudsdk.third_party.appengine.datastore.entity_v4_pb
def clear_index_updates(self):
    if self.has_index_updates_:
        self.has_index_updates_ = 0
        self.index_updates_ = 0