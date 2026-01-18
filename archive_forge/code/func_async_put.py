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
def async_put(self, config, entities, extra_hook=None):
    """Transactional asynchronous Put operation.

    Args:
      config: A Configuration object or None.  Defaults are taken from
        the connection's default configuration.
      entities: An iterable of user-level entity objects.
      extra_hook: Optional function to be called on the result once the
        RPC has completed.

     Returns:
      A MultiRpc object.

    NOTE: If any of the entities has an incomplete key, this will
    *not* patch up those entities with the complete key.
    """
    if self._api_version != _CLOUD_DATASTORE_V1:
        return super(TransactionalConnection, self).async_put(config, entities, extra_hook)
    v1_entities = [self.adapter.entity_to_pb_v1(entity) for entity in entities]
    v1_req = googledatastore.AllocateIdsRequest()
    for v1_entity in v1_entities:
        if not datastore_pbs.is_complete_v1_key(v1_entity.key):
            v1_req.keys.add().CopyFrom(v1_entity.key)
    user_data = (v1_entities, extra_hook)
    service_name = _CLOUD_DATASTORE_V1
    if not v1_req.keys:
        service_name = _NOOP_SERVICE
    return self._make_rpc_call(config, 'AllocateIds', v1_req, googledatastore.AllocateIdsResponse(), get_result_hook=self.__v1_put_allocate_ids_hook, user_data=user_data, service_name=service_name)