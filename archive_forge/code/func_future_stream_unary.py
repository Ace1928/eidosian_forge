import abc
import collections
import enum
from grpc.framework.common import cardinality  # pylint: disable=unused-import
from grpc.framework.common import style  # pylint: disable=unused-import
from grpc.framework.foundation import future  # pylint: disable=unused-import
from grpc.framework.foundation import stream  # pylint: disable=unused-import
@abc.abstractmethod
def future_stream_unary(self, group, method, request_iterator, timeout, metadata=None, protocol_options=None):
    """Invokes a stream-request-unary-response method.

        Args:
          group: The group identifier of the RPC.
          method: The method identifier of the RPC.
          request_iterator: An iterator that yields request values for the RPC.
          timeout: A duration of time in seconds to allow for the RPC.
          metadata: A metadata value to be passed to the service-side of the RPC.
          protocol_options: A value specified by the provider of a Face interface
            implementation affording custom state and behavior.

        Returns:
          An object that is both a Call for the RPC and a future.Future. In the
            event of RPC completion, the return Future's result value will be the
            response value of the RPC. In the event of RPC abortion, the returned
            Future's exception value will be an AbortionError.
        """
    raise NotImplementedError()