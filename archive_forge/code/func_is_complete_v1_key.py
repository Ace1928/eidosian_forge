from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def is_complete_v1_key(v1_key):
    """Returns True if a key specifies an ID or name, False otherwise.

  Args:
    v1_key: an googledatastore.Key

  Returns:
    True if the key specifies an ID or name, False otherwise.
  """
    assert len(v1_key.path) >= 1
    last_element = v1_key.path[len(v1_key.path) - 1]
    return last_element.WhichOneof('id_type') is not None