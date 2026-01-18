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
def projection(value):
    """A list or tuple of property names to project.

    If None, the entire entity is returned.

    Specifying a projection:
    - may change the index requirements for the given query;
    - will cause a partial entity to be returned;
    - will cause only entities that contain those properties to be returned;

    A partial entities only contain the property name and value for properties
    in the projection (meaning and multiple will not be set). They will also
    only contain a single value for any multi-valued property. However, if a
    multi-valued property is specified in the order, an inequality property, or
    the projected properties, the entity will be returned multiple times. Once
    for each unique combination of values.

    However, projection queries are significantly faster than normal queries.

    Raises:
      datastore_errors.BadArgumentError if value is empty or not a list or tuple
    of strings.
    """
    if isinstance(value, list):
        value = tuple(value)
    elif not isinstance(value, tuple):
        raise datastore_errors.BadArgumentError('projection argument should be a list or tuple (%r)' % (value,))
    if not value:
        raise datastore_errors.BadArgumentError('projection argument cannot be empty')
    for prop in value:
        if not isinstance(prop, six_subset.string_types + (six_subset.binary_type,)):
            raise datastore_errors.BadArgumentError('projection argument should contain only strings (%r)' % (prop,))
    return value