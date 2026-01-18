from __future__ import annotations
import inspect
import json
import re
import struct
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import wraps
from itertools import islice
from pathlib import Path
from typing import (
from urllib.parse import quote
import requests
from requests.exceptions import HTTPError
from tqdm.auto import tqdm as base_tqdm
from tqdm.contrib.concurrent import thread_map
from ._commit_api import (
from ._inference_endpoints import InferenceEndpoint, InferenceEndpointType
from ._multi_commits import (
from ._space_api import SpaceHardware, SpaceRuntime, SpaceStorage, SpaceVariable
from .community import (
from .constants import (
from .file_download import HfFileMetadata, get_hf_file_metadata, hf_hub_url
from .repocard_data import DatasetCardData, ModelCardData, SpaceCardData
from .utils import (  # noqa: F401 # imported for backward compatibility
from .utils import tqdm as hf_tqdm
from .utils._deprecation import _deprecate_arguments, _deprecate_method
from .utils._typing import CallableT
from .utils.endpoint_helpers import (
def create_inference_endpoint(self, name: str, *, repository: str, framework: str, accelerator: str, instance_size: str, instance_type: str, region: str, vendor: str, account_id: Optional[str]=None, min_replica: int=0, max_replica: int=1, revision: Optional[str]=None, task: Optional[str]=None, custom_image: Optional[Dict]=None, type: InferenceEndpointType=InferenceEndpointType.PROTECTED, namespace: Optional[str]=None, token: Optional[str]=None) -> InferenceEndpoint:
    """Create a new Inference Endpoint.

        Args:
            name (`str`):
                The unique name for the new Inference Endpoint.
            repository (`str`):
                The name of the model repository associated with the Inference Endpoint (e.g. `"gpt2"`).
            framework (`str`):
                The machine learning framework used for the model (e.g. `"custom"`).
            accelerator (`str`):
                The hardware accelerator to be used for inference (e.g. `"cpu"`).
            instance_size (`str`):
                The size or type of the instance to be used for hosting the model (e.g. `"large"`).
            instance_type (`str`):
                The cloud instance type where the Inference Endpoint will be deployed (e.g. `"c6i"`).
            region (`str`):
                The cloud region in which the Inference Endpoint will be created (e.g. `"us-east-1"`).
            vendor (`str`):
                The cloud provider or vendor where the Inference Endpoint will be hosted (e.g. `"aws"`).
            account_id (`str`, *optional*):
                The account ID used to link a VPC to a private Inference Endpoint (if applicable).
            min_replica (`int`, *optional*):
                The minimum number of replicas (instances) to keep running for the Inference Endpoint. Defaults to 0.
            max_replica (`int`, *optional*):
                The maximum number of replicas (instances) to scale to for the Inference Endpoint. Defaults to 1.
            revision (`str`, *optional*):
                The specific model revision to deploy on the Inference Endpoint (e.g. `"6c0e6080953db56375760c0471a8c5f2929baf11"`).
            task (`str`, *optional*):
                The task on which to deploy the model (e.g. `"text-classification"`).
            custom_image (`Dict`, *optional*):
                A custom Docker image to use for the Inference Endpoint. This is useful if you want to deploy an
                Inference Endpoint running on the `text-generation-inference` (TGI) framework (see examples).
            type ([`InferenceEndpointType]`, *optional*):
                The type of the Inference Endpoint, which can be `"protected"` (default), `"public"` or `"private"`.
            namespace (`str`, *optional*):
                The namespace where the Inference Endpoint will be created. Defaults to the current user's namespace.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token).

            Returns:
                [`InferenceEndpoint`]: information about the updated Inference Endpoint.

            Example:
            ```python
            >>> from huggingface_hub import HfApi
            >>> api = HfApi()
            >>> create_inference_endpoint(
            ...     "my-endpoint-name",
            ...     repository="gpt2",
            ...     framework="pytorch",
            ...     task="text-generation",
            ...     accelerator="cpu",
            ...     vendor="aws",
            ...     region="us-east-1",
            ...     type="protected",
            ...     instance_size="medium",
            ...     instance_type="c6i",
            ... )
            >>> endpoint
            InferenceEndpoint(name='my-endpoint-name', status="pending",...)

            # Run inference on the endpoint
            >>> endpoint.client.text_generation(...)
            "..."
            ```

            ```python
            # Start an Inference Endpoint running Zephyr-7b-beta on TGI
            >>> from huggingface_hub import HfApi
            >>> api = HfApi()
            >>> create_inference_endpoint(
            ...     "aws-zephyr-7b-beta-0486",
            ...     repository="HuggingFaceH4/zephyr-7b-beta",
            ...     framework="pytorch",
            ...     task="text-generation",
            ...     accelerator="gpu",
            ...     vendor="aws",
            ...     region="us-east-1",
            ...     type="protected",
            ...     instance_size="medium",
            ...     instance_type="g5.2xlarge",
            ...     custom_image={
            ...         "health_route": "/health",
            ...         "env": {
            ...             "MAX_BATCH_PREFILL_TOKENS": "2048",
            ...             "MAX_INPUT_LENGTH": "1024",
            ...             "MAX_TOTAL_TOKENS": "1512",
            ...             "MODEL_ID": "/repository"
            ...         },
            ...         "url": "ghcr.io/huggingface/text-generation-inference:1.1.0",
            ...     },
            ... )

            ```
        """
    namespace = namespace or self._get_namespace(token=token)
    image = {'custom': custom_image} if custom_image is not None else {'huggingface': {}}
    payload: Dict = {'accountId': account_id, 'compute': {'accelerator': accelerator, 'instanceSize': instance_size, 'instanceType': instance_type, 'scaling': {'maxReplica': max_replica, 'minReplica': min_replica}}, 'model': {'framework': framework, 'repository': repository, 'revision': revision, 'task': task, 'image': image}, 'name': name, 'provider': {'region': region, 'vendor': vendor}, 'type': type}
    response = get_session().post(f'{INFERENCE_ENDPOINTS_ENDPOINT}/endpoint/{namespace}', headers=self._build_hf_headers(token=token), json=payload)
    hf_raise_for_status(response)
    return InferenceEndpoint.from_raw(response.json(), namespace=namespace, token=token)