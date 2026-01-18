import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Dict, Optional
from .inference._client import InferenceClient
from .inference._generated._async_client import AsyncInferenceClient
from .utils import logging, parse_datetime
def scale_to_zero(self) -> 'InferenceEndpoint':
    """Scale Inference Endpoint to zero.

        An Inference Endpoint scaled to zero will not be charged. It will be resume on the next request to it, with a
        cold start delay. This is different than pausing the Inference Endpoint with [`InferenceEndpoint.pause`], which
        would require a manual resume with [`InferenceEndpoint.resume`].

        This is an alias for [`HfApi.scale_to_zero_inference_endpoint`]. The current object is mutated in place with the
        latest data from the server.

        Returns:
            [`InferenceEndpoint`]: the same Inference Endpoint, mutated in place with the latest data.
        """
    obj = self._api.scale_to_zero_inference_endpoint(name=self.name, namespace=self.namespace, token=self._token)
    self.raw = obj.raw
    self._populate_from_raw()
    return self