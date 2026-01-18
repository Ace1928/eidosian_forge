import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Dict, Optional
from .inference._client import InferenceClient
from .inference._generated._async_client import AsyncInferenceClient
from .utils import logging, parse_datetime
@property
def async_client(self) -> AsyncInferenceClient:
    """Returns a client to make predictions on this Inference Endpoint.

        Returns:
            [`AsyncInferenceClient`]: an asyncio-compatible inference client pointing to the deployed endpoint.

        Raises:
            [`InferenceEndpointError`]: If the Inference Endpoint is not yet deployed.
        """
    if self.url is None:
        raise InferenceEndpointError('Cannot create a client for this Inference Endpoint as it is not yet deployed. Please wait for the Inference Endpoint to be deployed using `endpoint.wait()` and try again.')
    return AsyncInferenceClient(model=self.url, token=self._token)