from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.entity_v4_pb import *
import googlecloudsdk.third_party.appengine.datastore.entity_v4_pb
def read_options(self):
    if self.read_options_ is None:
        self.lazy_init_lock_.acquire()
        try:
            if self.read_options_ is None:
                self.read_options_ = ReadOptions()
        finally:
            self.lazy_init_lock_.release()
    return self.read_options_