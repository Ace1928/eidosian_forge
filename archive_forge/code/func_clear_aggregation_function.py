from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.entity_v4_pb import *
import googlecloudsdk.third_party.appengine.datastore.entity_v4_pb
def clear_aggregation_function(self):
    if self.has_aggregation_function_:
        self.has_aggregation_function_ = 0
        self.aggregation_function_ = 0