from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.entity_v4_pb import *
import googlecloudsdk.third_party.appengine.datastore.entity_v4_pb
def clear_ignore_read_only(self):
    if self.has_ignore_read_only_:
        self.has_ignore_read_only_ = 0
        self.ignore_read_only_ = 0