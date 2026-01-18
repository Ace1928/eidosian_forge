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
def async_commit(self, config):
    """Asynchronous Commit operation.

    Args:
      config: A Configuration object or None.  Defaults are taken from
        the connection's default configuration.

    Returns:
      A MultiRpc object.
    """
    self.wait_for_all_pending_rpcs()
    if self._state != TransactionalConnection.OPEN:
        raise datastore_errors.BadRequestError('Transaction is already finished.')
    self._state = TransactionalConnection.COMMIT_IN_FLIGHT
    transaction = self.transaction
    if transaction is None:
        self._state = TransactionalConnection.CLOSED
        return None
    if self._api_version == _CLOUD_DATASTORE_V1:
        req = googledatastore.CommitRequest()
        req.transaction = transaction
        if Configuration.force_writes(config, self.__config):
            self.__force(req)
        for entity in self.__pending_v1_upserts.values():
            mutation = req.mutations.add()
            mutation.upsert.CopyFrom(entity)
        for key in self.__pending_v1_deletes.values():
            mutation = req.mutations.add()
            mutation.delete.CopyFrom(key)
        self.__pending_v1_upserts.clear()
        self.__pending_v1_deletes.clear()
        resp = googledatastore.CommitResponse()
    else:
        req = transaction
        resp = datastore_pb.CommitResponse()
    return self._make_rpc_call(config, 'Commit', req, resp, get_result_hook=self.__commit_hook, service_name=self._api_version)