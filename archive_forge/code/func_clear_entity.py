from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.entity_v4_pb import *
import googlecloudsdk.third_party.appengine.datastore.entity_v4_pb
def clear_entity(self):
    if self.has_entity_:
        self.has_entity_ = 0
        if self.entity_ is not None:
            self.entity_.Clear()