import abc
import collections
import enum
from grpc.framework.common import cardinality  # pylint: disable=unused-import
from grpc.framework.common import style  # pylint: disable=unused-import
from grpc.framework.foundation import future  # pylint: disable=unused-import
from grpc.framework.foundation import stream  # pylint: disable=unused-import
@abc.abstractmethod
def terminal_metadata(self, terminal_metadata):
    """Accepts the service-side terminal metadata value of the RPC.

        This method need not be called by method implementations if they have no
        service-side terminal metadata to transmit.

        Args:
          terminal_metadata: The service-side terminal metadata value of the RPC to
            be transmitted to the invocation side of the RPC.
        """
    raise NotImplementedError()