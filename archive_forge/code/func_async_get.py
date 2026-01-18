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
def async_get(self, config, keys, extra_hook=None):
    """Asynchronous Get operation.

    Args:
      config: A Configuration object or None.  Defaults are taken from
        the connection's default configuration.
      keys: An iterable of user-level key objects.
      extra_hook: Optional function to be called on the result once the
        RPC has completed.

    Returns:
      A MultiRpc object.
    """

    def make_get_call(base_req, pbs, extra_hook=None):
        req = copy.deepcopy(base_req)
        if self._api_version == _CLOUD_DATASTORE_V1:
            method = 'Lookup'
            req.keys.extend(pbs)
            resp = googledatastore.LookupResponse()
        else:
            method = 'Get'
            req.key_list().extend(pbs)
            resp = datastore_pb.GetResponse()
        user_data = (config, pbs, extra_hook)
        return self._make_rpc_call(config, method, req, resp, get_result_hook=self.__get_hook, user_data=user_data, service_name=self._api_version)
    if self._api_version == _CLOUD_DATASTORE_V1:
        base_req = googledatastore.LookupRequest()
        key_to_pb = self.__adapter.key_to_pb_v1
    else:
        base_req = datastore_pb.GetRequest()
        base_req.set_allow_deferred(True)
        key_to_pb = self.__adapter.key_to_pb
    is_read_current = self._set_request_read_policy(base_req, config)
    txn = self._set_request_transaction(base_req)
    if isinstance(config, apiproxy_stub_map.UserRPC) or len(keys) <= 1:
        pbs = [key_to_pb(key) for key in keys]
        return make_get_call(base_req, pbs, extra_hook)
    max_count = Configuration.max_get_keys(config, self.__config) or self.MAX_GET_KEYS
    indexed_keys_by_entity_group = self._map_and_group(keys, key_to_pb, self._extract_entity_group)
    if is_read_current is None:
        is_read_current = self.get_datastore_type() == BaseConnection.HIGH_REPLICATION_DATASTORE
    if is_read_current and txn is None:
        max_egs_per_rpc = self.__get_max_entity_groups_per_rpc(config)
    else:
        max_egs_per_rpc = None
    pbsgen = self._generate_pb_lists(indexed_keys_by_entity_group, base_req.ByteSize(), max_count, max_egs_per_rpc, config)
    rpcs = []
    for pbs, indexes in pbsgen:
        rpcs.append(make_get_call(base_req, pbs, self.__create_result_index_pairs(indexes)))
    return MultiRpc(rpcs, self.__sort_result_index_pairs(extra_hook))