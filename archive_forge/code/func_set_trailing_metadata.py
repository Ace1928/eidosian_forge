from typing import Any, Dict, List, Optional, Tuple
import grpc
from ray.util.annotations import DeveloperAPI, PublicAPI
def set_trailing_metadata(self, trailing_metadata: List[Tuple[str, str]]):
    """Sets the trailing metadata for the RPC.

        Sets the trailing metadata to be sent upon completion of the RPC.

        If this method is invoked multiple times throughout the lifetime of an
        RPC, the value supplied in the final invocation + request id will be the value
        sent over the wire.

        This method need not be called by implementations if they have no
        metadata to add to what the gRPC runtime will transmit.

        Args:
          trailing_metadata: The trailing :term:`metadata`.
        """
    self._trailing_metadata = self._request_id_metadata() + trailing_metadata