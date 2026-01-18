from __future__ import absolute_import
from __future__ import unicode_literals
import base64
import collections
import pickle
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine._internal import six_subset
from googlecloudsdk.third_party.appengine.api import datastore_errors
from googlecloudsdk.third_party.appengine.api import datastore_types
from googlecloudsdk.third_party.appengine.datastore import datastore_index
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_pbs
from googlecloudsdk.third_party.appengine.datastore import datastore_rpc
@classmethod
def from_property_filter(cls, prop_filter):
    op = prop_filter._filter.op()
    if op == datastore_pb.Query_Filter.GREATER_THAN:
        return cls(start=prop_filter._filter.property(0), start_incl=False)
    elif op == datastore_pb.Query_Filter.GREATER_THAN_OR_EQUAL:
        return cls(start=prop_filter._filter.property(0))
    elif op == datastore_pb.Query_Filter.LESS_THAN:
        return cls(end=prop_filter._filter.property(0), end_incl=False)
    elif op == datastore_pb.Query_Filter.LESS_THAN_OR_EQUAL:
        return cls(end=prop_filter._filter.property(0))
    else:
        raise datastore_errors.BadArgumentError('Unsupported operator (%s)' % (op,))