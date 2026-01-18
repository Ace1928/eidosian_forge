from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def entity_value(self):
    if self.entity_value_ is None:
        self.lazy_init_lock_.acquire()
        try:
            if self.entity_value_ is None:
                self.entity_value_ = Entity()
        finally:
            self.lazy_init_lock_.release()
    return self.entity_value_