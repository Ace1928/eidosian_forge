from typing import Any, Dict, List, Optional, Tuple
import grpc
from ray.util.annotations import DeveloperAPI, PublicAPI
@DeveloperAPI
def set_on_grpc_context(self, grpc_context: grpc._cython.cygrpc._ServicerContext):
    """Serve's internal method to set attributes on the gRPC context."""
    if self._code:
        grpc_context.set_code(self._code)
    if self._compression:
        grpc_context.set_compression(self._compression)
    if self._details:
        grpc_context.set_details(self._details)
    if self._trailing_metadata:
        grpc_context.set_trailing_metadata(self._trailing_metadata)