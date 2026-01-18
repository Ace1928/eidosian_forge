from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def get_v1_mutation_key_and_entity(v1_mutation):
    """Returns the v1 key and entity for a v1 mutation proto, if applicable.

  Args:
    v1_mutation: a googledatastore.Mutation

  Returns:
    a tuple (googledatastore.Key for this mutation,
             googledatastore.Entity or None if the mutation is a deletion)
  """
    if v1_mutation.HasField('delete'):
        return (v1_mutation.delete, None)
    else:
        v1_entity = getattr(v1_mutation, v1_mutation.WhichOneof('operation'))
        return (v1_entity.key, v1_entity)