from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.entity_v4_pb import *
import googlecloudsdk.third_party.appengine.datastore.entity_v4_pb
def deprecated_mutation_result(self):
    if self.deprecated_mutation_result_ is None:
        self.lazy_init_lock_.acquire()
        try:
            if self.deprecated_mutation_result_ is None:
                self.deprecated_mutation_result_ = DeprecatedMutationResult()
        finally:
            self.lazy_init_lock_.release()
    return self.deprecated_mutation_result_