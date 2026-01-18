from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb import *
import googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb
from googlecloudsdk.third_party.appengine.proto.message_set import MessageSet
def cron_retry_parameters(self):
    if self.cron_retry_parameters_ is None:
        self.lazy_init_lock_.acquire()
        try:
            if self.cron_retry_parameters_ is None:
                self.cron_retry_parameters_ = TaskQueueRetryParameters()
        finally:
            self.lazy_init_lock_.release()
    return self.cron_retry_parameters_