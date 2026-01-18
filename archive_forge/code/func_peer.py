from typing import Any, Dict, List, Optional, Tuple
import grpc
from ray.util.annotations import DeveloperAPI, PublicAPI
def peer(self) -> str:
    """Identifies the peer that invoked the RPC being serviced.

        Returns:
          A string identifying the peer that invoked the RPC being serviced.
          The string format is determined by gRPC runtime.
        """
    return self._peer