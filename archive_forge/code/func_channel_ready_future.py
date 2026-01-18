import threading
import time
from grpc.beta import implementations  # pylint: disable=unused-import
from grpc.beta import interfaces
from grpc.framework.foundation import callable_util
from grpc.framework.foundation import future
def channel_ready_future(channel):
    """Creates a future.Future tracking when an implementations.Channel is ready.

    Cancelling the returned future.Future does not tell the given
    implementations.Channel to abandon attempts it may have been making to
    connect; cancelling merely deactivates the return future.Future's
    subscription to the given implementations.Channel's connectivity.

    Args:
      channel: An implementations.Channel.

    Returns:
      A future.Future that matures when the given Channel has connectivity
        interfaces.ChannelConnectivity.READY.
    """
    ready_future = _ChannelReadyFuture(channel)
    ready_future.start()
    return ready_future