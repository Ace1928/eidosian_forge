from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict
import aiohttp
from mlflow.gateway.constants import (
from mlflow.utils.uri import append_to_uri_path
def rename_payload_keys(payload: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
    """Rename payload keys based on the specified mapping. If a key is not present in the
    mapping, the key and its value will remain unchanged.

    Args:
        payload: The original dictionary to transform.
        mapping: A dictionary where each key-value pair represents a mapping from the old
            key to the new key.

    Returns:
        A new dictionary containing the transformed keys.

    """
    return {mapping.get(k, k): v for k, v in payload.items()}