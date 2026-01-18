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
def allocate_ids(self, key, size=None, max=None):
    """Synchronous AllocateIds operation.

    Exactly one of size and max must be specified.

    Args:
      key: A user-level key object.
      size: Optional number of IDs to allocate.
      max: Optional maximum ID to allocate.

    Returns:
      A pair (start, end) giving the (inclusive) range of IDs allocation.
    """
    return self.async_allocate_ids(None, key, size, max).get_result()