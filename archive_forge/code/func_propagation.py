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
@ConfigOption
def propagation(value):
    """How existing transactions should be handled.

    One of NESTED, MANDATORY, ALLOWED, INDEPENDENT. The interpertation of
    these types is up to higher level run-in-transaction implementations.

    WARNING: Using anything other than NESTED for the propagation flag
    can have strange consequences.  When using ALLOWED or MANDATORY, if
    an exception is raised, the transaction is likely not safe to
    commit.  When using INDEPENDENT it is not generally safe to return
    values read to the caller (as they were not read in the caller's
    transaction).

    Raises: datastore_errors.BadArgumentError if value is not reconized.
    """
    if value not in TransactionOptions._PROPAGATION:
        raise datastore_errors.BadArgumentError('Unknown propagation value (%r)' % (value,))
    return value