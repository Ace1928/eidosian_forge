from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def is_complete_v3_key(v3_key):
    """Returns True if a key specifies an ID or name, False otherwise.

  Args:
    v3_key: a datastore_pb.Reference

  Returns:
    True if the key specifies an ID or name, False otherwise.
  """
    assert v3_key.path().element_size() >= 1
    last_element = v3_key.path().element_list()[-1]
    return last_element.has_id() and last_element.id() != 0 or (last_element.has_name() and last_element.name() != '')