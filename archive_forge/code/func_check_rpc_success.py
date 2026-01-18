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
def check_rpc_success(self, rpc):
    """Check for RPC success and translate exceptions.

    This wraps rpc.check_success() and should be called instead of that.

    This also removes the RPC from the list of pending RPCs, once it
    has completed.

    Args:
      rpc: A UserRPC or MultiRpc object.

    Raises:
      Nothing if the call succeeded; various datastore_errors.Error
      subclasses if ApplicationError was raised by rpc.check_success().
    """
    try:
        rpc.wait()
    finally:
        self._remove_pending(rpc)
    try:
        rpc.check_success()
    except apiproxy_errors.ApplicationError as err:
        raise _ToDatastoreError(err)