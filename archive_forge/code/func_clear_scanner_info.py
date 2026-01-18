from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb import *
import googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb
from googlecloudsdk.third_party.appengine.proto.message_set import MessageSet
def clear_scanner_info(self):
    if self.has_scanner_info_:
        self.has_scanner_info_ = 0
        if self.scanner_info_ is not None:
            self.scanner_info_.Clear()