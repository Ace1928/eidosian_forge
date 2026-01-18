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
def pb_v1_to_key(self, pb):
    """Turn an googledatastore.Key into a user-level key."""
    v3_ref = entity_pb.Reference()
    self._entity_converter.v1_to_v3_reference(pb, v3_ref)
    return self.pb_to_key(v3_ref)