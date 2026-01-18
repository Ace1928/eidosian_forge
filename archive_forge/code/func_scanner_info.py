from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb import *
import googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb
from googlecloudsdk.third_party.appengine.proto.message_set import MessageSet
def scanner_info(self):
    if self.scanner_info_ is None:
        self.lazy_init_lock_.acquire()
        try:
            if self.scanner_info_ is None:
                self.scanner_info_ = TaskQueueScannerQueueInfo()
        finally:
            self.lazy_init_lock_.release()
    return self.scanner_info_