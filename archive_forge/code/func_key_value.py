from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def key_value(self):
    if self.key_value_ is None:
        self.lazy_init_lock_.acquire()
        try:
            if self.key_value_ is None:
                self.key_value_ = Key()
        finally:
            self.lazy_init_lock_.release()
    return self.key_value_