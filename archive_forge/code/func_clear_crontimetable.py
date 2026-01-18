from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb import *
import googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb
from googlecloudsdk.third_party.appengine.proto.message_set import MessageSet
def clear_crontimetable(self):
    if self.has_crontimetable_:
        self.has_crontimetable_ = 0
        if self.crontimetable_ is not None:
            self.crontimetable_.Clear()