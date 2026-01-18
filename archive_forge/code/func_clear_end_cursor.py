from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.entity_v4_pb import *
import googlecloudsdk.third_party.appengine.datastore.entity_v4_pb
def clear_end_cursor(self):
    if self.has_end_cursor_:
        self.has_end_cursor_ = 0
        self.end_cursor_ = ''