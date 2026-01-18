from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.entity_v4_pb import *
import googlecloudsdk.third_party.appengine.datastore.entity_v4_pb
def clear_skipped_cursor(self):
    if self.has_skipped_cursor_:
        self.has_skipped_cursor_ = 0
        self.skipped_cursor_ = ''