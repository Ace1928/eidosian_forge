from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb import *
import googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb
from googlecloudsdk.third_party.appengine.proto.message_set import MessageSet
def clear_min_backoff_sec(self):
    if self.has_min_backoff_sec_:
        self.has_min_backoff_sec_ = 0
        self.min_backoff_sec_ = 0.1