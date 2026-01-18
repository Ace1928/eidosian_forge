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
@datastore_rpc._positional(5)
def create_async(cls, augmented_query, query_options, conn, req, in_memory_offset, in_memory_limit, start_cursor):
    initial_offset = 0 if in_memory_offset is not None else None
    batch_shared = _BatchShared(augmented_query._query, query_options, conn, augmented_query, initial_offset=initial_offset)
    batch0 = cls(batch_shared, in_memory_offset=in_memory_offset, in_memory_limit=in_memory_limit, start_cursor=start_cursor)
    return batch0._make_query_rpc_call(query_options, req)