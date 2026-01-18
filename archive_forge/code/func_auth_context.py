from typing import Any, Dict, List, Optional, Tuple
import grpc
from ray.util.annotations import DeveloperAPI, PublicAPI
def auth_context(self) -> Dict[str, Any]:
    """Gets the auth context for the call.

        Returns:
          A map of strings to an iterable of bytes for each auth property.
        """
    return self._auth_context