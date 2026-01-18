from __future__ import absolute_import
from __future__ import unicode_literals
import collections
import copy
import functools
import logging
import os
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine._internal import six_subset
from googlecloudsdk.third_party.appengine.api import api_base_pb
from googlecloudsdk.third_party.appengine.api import apiproxy_rpc
from googlecloudsdk.third_party.appengine.api import apiproxy_stub_map
from googlecloudsdk.third_party.appengine.api import datastore_errors
from googlecloudsdk.third_party.appengine.api import datastore_types
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_pbs
from googlecloudsdk.third_party.appengine.runtime import apiproxy_errors
def async_allocate_ids(self, config, key, size=None, max=None, extra_hook=None):
    """Asynchronous AllocateIds operation.

    Args:
      config: A Configuration object or None.  Defaults are taken from
        the connection's default configuration.
      key: A user-level key object.
      size: Optional number of IDs to allocate.
      max: Optional maximum ID to allocate.
      extra_hook: Optional function to be called on the result once the
        RPC has completed.

    Returns:
      A MultiRpc object.
    """
    if size is not None:
        if max is not None:
            raise datastore_errors.BadArgumentError('Cannot allocate ids using both size and max')
        if not isinstance(size, six_subset.integer_types):
            raise datastore_errors.BadArgumentError('Invalid size (%r)' % (size,))
        if size > _MAX_ID_BATCH_SIZE:
            raise datastore_errors.BadArgumentError('Cannot allocate more than %s ids at a time; received %s' % (_MAX_ID_BATCH_SIZE, size))
        if size <= 0:
            raise datastore_errors.BadArgumentError('Cannot allocate less than 1 id; received %s' % size)
    if max is not None:
        if not isinstance(max, six_subset.integer_types):
            raise datastore_errors.BadArgumentError('Invalid max (%r)' % (max,))
        if max < 0:
            raise datastore_errors.BadArgumentError('Cannot allocate a range with a max less than 0 id; received %s' % size)
    req = datastore_pb.AllocateIdsRequest()
    req.mutable_model_key().CopyFrom(self.__adapter.key_to_pb(key))
    if size is not None:
        req.set_size(size)
    if max is not None:
        req.set_max(max)
    resp = datastore_pb.AllocateIdsResponse()
    rpc = self._make_rpc_call(config, 'AllocateIds', req, resp, get_result_hook=self.__allocate_ids_hook, user_data=extra_hook, service_name=_DATASTORE_V3)
    return rpc