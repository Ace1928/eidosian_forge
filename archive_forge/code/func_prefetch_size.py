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
@datastore_rpc.ConfigOption
def prefetch_size(value):
    """Number of results to attempt to return on the initial request.

    Raises:
      datastore_errors.BadArgumentError if value is not an integer or is not
      greater than zero.
    """
    datastore_types.ValidateInteger(value, 'prefetch_size', datastore_errors.BadArgumentError, zero_ok=True)
    return value