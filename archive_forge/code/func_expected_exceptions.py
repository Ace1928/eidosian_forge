import logging
import sys
from oslo_messaging import exceptions
from oslo_messaging.rpc import dispatcher as rpc_dispatcher
from oslo_messaging import server as msg_server
from oslo_messaging import transport as msg_transport
def expected_exceptions(*exceptions):
    """Decorator for RPC endpoint methods that raise expected exceptions.

    Marking an endpoint method with this decorator allows the declaration
    of expected exceptions that the RPC server should not consider fatal,
    and not log as if they were generated in a real error scenario.

    Note that this will cause listed exceptions to be wrapped in an
    ExpectedException, which is used internally by the RPC sever. The RPC
    client will see the original exception type.
    """

    def outer(func):

        def inner(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions:
                raise rpc_dispatcher.ExpectedException()
        return inner
    return outer