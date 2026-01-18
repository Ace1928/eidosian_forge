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
def apply_query(query, entities, _key=None):
    """Performs the given query on a set of in-memory results.

  This function can perform queries impossible in the datastore (e.g a query
  with multiple inequality filters on different properties) because all
  operations are done in memory. For queries that can also be executed on the
  the datastore, the results produced by this function may not use the same
  implicit ordering as the datastore. To ensure compatibility, explicit
  ordering must be used (e.g. 'ORDER BY ineq_prop, ..., __key__').

  Order by __key__ should always be used when a consistent result is desired
  (unless there is a sort order on another globally unique property).

  Args:
    query: a datastore_query.Query to apply
    entities: a list of results, of arbitrary type, on which to apply the query.
    _key: a function that takes an element of the result array as an argument
        and must return an entity_pb.EntityProto. If not specified, the identity
        function is used (and entities must be a list of entity_pb.EntityProto).

  Returns:
    A subset of entities, filtered and ordered according to the query.
  """
    if not isinstance(query, Query):
        raise datastore_errors.BadArgumentError('query argument must be a datastore_query.Query (%r)' % (query,))
    if not isinstance(entities, list):
        raise datastore_errors.BadArgumentError('entities argument must be a list (%r)' % (entities,))
    key = _key or (lambda x: x)
    filtered_results = [r for r in entities if query._key_filter(key(r))]
    if not query._order:
        if query._filter_predicate:
            return [r for r in filtered_results if query._filter_predicate(key(r))]
        return filtered_results
    names = query._order._get_prop_names()
    if query._filter_predicate:
        names |= query._filter_predicate._get_prop_names()
    exists_filter = _PropertyExistsFilter(names)
    value_maps = []
    for result in filtered_results:
        value_map = _make_key_value_map(key(result), names)
        if exists_filter._apply(value_map) and (not query._filter_predicate or query._filter_predicate._prune(value_map)):
            value_map['__result__'] = result
            value_maps.append(value_map)
    value_maps.sort(query._order._cmp)
    return [value_map['__result__'] for value_map in value_maps]