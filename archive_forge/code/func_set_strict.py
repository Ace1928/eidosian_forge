from __future__ import absolute_import
import io
import logging
from googlecloudsdk.third_party.appengine.tools.appengine_rpc import AbstractRpcServer
from googlecloudsdk.third_party.appengine.tools.appengine_rpc import HttpRpcServer
from googlecloudsdk.third_party.appengine._internal import six_subset
def set_strict(self, strict=True):
    """Enables strict mode."""
    self.strict = strict