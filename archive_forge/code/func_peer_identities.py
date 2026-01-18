from typing import Any, Dict, List, Optional, Tuple
import grpc
from ray.util.annotations import DeveloperAPI, PublicAPI
def peer_identities(self) -> Optional[bytes]:
    """Gets one or more peer identity(s).

        Equivalent to
        servicer_context.auth_context().get(servicer_context.peer_identity_key())

        Returns:
          An iterable of the identities, or None if the call is not
          authenticated. Each identity is returned as a raw bytes type.
        """
    return self._peer_identities