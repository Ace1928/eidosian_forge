from __future__ import absolute_import
import inspect
import sys
import threading
from googlecloudsdk.third_party.appengine.api import apiproxy_rpc
from googlecloudsdk.third_party.appengine.runtime import apiproxy_errors
def make_call(self, method, request, response, get_result_hook=None, user_data=None):
    """Initiate a call.

    Args:
      method: The method name.
      request: The request protocol buffer.
      response: The response protocol buffer.
      get_result_hook: Optional get-result hook function.  If not None,
        this must be a function with exactly one argument, the RPC
        object (self).  Its return value is returned from get_result().
      user_data: Optional additional arbitrary data for the get-result
        hook function.  This can be accessed as rpc.user_data.  The
        type of this value is up to the service module.

    This function may only be called once per RPC object.  It sends
    the request to the remote server, but does not wait for a
    response.  This allows concurrent execution of the remote call and
    further local processing (e.g., making additional remote calls).

    Before the call is initiated, the precall hooks are called.
    """
    assert self.__rpc.state == apiproxy_rpc.RPC.IDLE, repr(self.state)
    self.__method = method
    self.__get_result_hook = get_result_hook
    self.__user_data = user_data
    self.__stubmap.GetPreCallHooks().Call(self.__service, method, request, response, self.__rpc)
    self.__rpc.MakeCall(self.__service, method, request, response)