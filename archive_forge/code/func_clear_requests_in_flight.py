from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb import *
import googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb
from googlecloudsdk.third_party.appengine.proto.message_set import MessageSet
def clear_requests_in_flight(self):
    if self.has_requests_in_flight_:
        self.has_requests_in_flight_ = 0
        self.requests_in_flight_ = 0