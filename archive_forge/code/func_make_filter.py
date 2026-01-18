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
def make_filter(name, op, values):
    """Constructs a FilterPredicate from the given name, op and values.

  Args:
    name: A non-empty string, the name of the property to filter.
    op: One of PropertyFilter._OPERATORS.keys(), the operator to use.
    values: A supported value, the value to compare against.

  Returns:
    if values is a list, a CompositeFilter that uses AND to combine all
    values, otherwise a PropertyFilter for the single value.

  Raises:
    datastore_errors.BadPropertyError: if the property name is invalid.
    datastore_errors.BadValueError: if the property did not validate correctly
      or the value was an empty list.
    Other exception types (like OverflowError): if the property value does not
      meet type-specific criteria.
  """
    datastore_types.ValidateProperty(name, values)
    properties = datastore_types.ToPropertyPb(name, values)
    if isinstance(properties, list):
        filters = [PropertyFilter(op, prop) for prop in properties]
        return CompositeFilter(CompositeFilter.AND, filters)
    else:
        return PropertyFilter(op, properties)