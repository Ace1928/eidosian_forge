from __future__ import absolute_import
import inspect
import sys
import threading
from googlecloudsdk.third_party.appengine.api import apiproxy_rpc
from googlecloudsdk.third_party.appengine.runtime import apiproxy_errors
@property
def get_result_hook(self):
    """Return the get-result hook function."""
    return self.__get_result_hook