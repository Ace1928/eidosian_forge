from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.entity_v4_pb import *
import googlecloudsdk.third_party.appengine.datastore.entity_v4_pb
def clear_read_consistency(self):
    if self.has_read_consistency_:
        self.has_read_consistency_ = 0
        self.read_consistency_ = 0