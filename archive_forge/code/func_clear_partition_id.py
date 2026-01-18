from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.entity_v4_pb import *
import googlecloudsdk.third_party.appengine.datastore.entity_v4_pb
def clear_partition_id(self):
    if self.has_partition_id_:
        self.has_partition_id_ = 0
        if self.partition_id_ is not None:
            self.partition_id_.Clear()