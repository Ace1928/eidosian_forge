from __future__ import absolute_import
import inspect
import sys
import threading
from googlecloudsdk.third_party.appengine.api import apiproxy_rpc
from googlecloudsdk.third_party.appengine.runtime import apiproxy_errors
@property
def deadline(self):
    """Return the deadline, if set explicitly (otherwise None)."""
    return self.__rpc.deadline