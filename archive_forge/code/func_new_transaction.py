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
def new_transaction(self, config=None, previous_transaction=None, mode=TransactionMode.UNKNOWN):
    """Create a new transactional connection based on this one.

    This is different from, and usually preferred over, the
    begin_transaction() method; new_transaction() returns a new
    TransactionalConnection object.

    Args:
      config: A configuration object for the new connection, merged
        with this connection's config.
      previous_transaction: The transaction being reset.
      mode: The transaction mode.
    """
    config = self.__config.merge(config)
    return TransactionalConnection(adapter=self.__adapter, config=config, _api_version=self._api_version, previous_transaction=previous_transaction, mode=mode)