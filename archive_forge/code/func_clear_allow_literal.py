from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.entity_v4_pb import *
import googlecloudsdk.third_party.appengine.datastore.entity_v4_pb
def clear_allow_literal(self):
    if self.has_allow_literal_:
        self.has_allow_literal_ = 0
        self.allow_literal_ = 0