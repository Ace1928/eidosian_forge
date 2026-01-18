from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb import *
import googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb
from googlecloudsdk.third_party.appengine.proto.message_set import MessageSet
def clear_dispatched_usec(self):
    if self.has_dispatched_usec_:
        self.has_dispatched_usec_ = 0
        self.dispatched_usec_ = 0