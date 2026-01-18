from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.entity_v4_pb import *
import googlecloudsdk.third_party.appengine.datastore.entity_v4_pb
def composite_filter(self):
    if self.composite_filter_ is None:
        self.lazy_init_lock_.acquire()
        try:
            if self.composite_filter_ is None:
                self.composite_filter_ = CompositeFilter()
        finally:
            self.lazy_init_lock_.release()
    return self.composite_filter_