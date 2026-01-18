from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.entity_v4_pb import *
import googlecloudsdk.third_party.appengine.datastore.entity_v4_pb
def clear_deprecated_mutation(self):
    if self.has_deprecated_mutation_:
        self.has_deprecated_mutation_ = 0
        if self.deprecated_mutation_ is not None:
            self.deprecated_mutation_.Clear()