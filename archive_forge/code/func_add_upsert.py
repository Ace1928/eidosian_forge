from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.entity_v4_pb import *
import googlecloudsdk.third_party.appengine.datastore.entity_v4_pb
def add_upsert(self):
    x = googlecloudsdk.third_party.appengine.datastore.entity_v4_pb.Entity()
    self.upsert_.append(x)
    return x