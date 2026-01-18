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
@staticmethod
def from_websafe_string(cursor):
    """Gets a Cursor given its websafe serialized form.

    The serialized form of a cursor may change in a non-backwards compatible
    way. In this case cursors must be regenerated from a new Query request.

    Args:
      cursor: A serialized cursor as returned by .to_websafe_string.

    Returns:
      A Cursor.

    Raises:
      datastore_errors.BadValueError if the cursor argument is not a string
      type of does not represent a serialized cursor.
    """
    decoded_bytes = Cursor._urlsafe_to_bytes(cursor)
    return Cursor.from_bytes(decoded_bytes)