import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, List, Optional, Sequence
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_core.utils.iter import batch_iterate
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_community.utilities.vertexai import get_client_info
def operations_from_names(self, operation_names: List[str]) -> List['Operation']:
    """Initializes Long-Running Operations from their names."""
    try:
        from google.longrunning.operations_pb2 import GetOperationRequest
    except ImportError as exc:
        raise ImportError('long running operations package not found, please install it with `pip install gapic-google-longrunning`') from exc
    return [self._client.get_operation(request=GetOperationRequest(name=name)) for name in operation_names]