from typing import Any, Dict, List, Optional, Tuple
import grpc
from ray.util.annotations import DeveloperAPI, PublicAPI
def set_compression(self, compression: grpc.Compression):
    """Set the compression algorithm to be used for the entire call.

        Args:
          compression: An element of grpc.compression, e.g.
            grpc.compression.Gzip.
        """
    self._compression = compression