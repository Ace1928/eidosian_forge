from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.utils import get_from_dict_or_env
from langchain_community.utilities.vertexai import get_client_info
@property
def client_options(self) -> 'ClientOptions':
    from google.api_core.client_options import ClientOptions
    return ClientOptions(api_endpoint=f'{self.location_id}-discoveryengine.googleapis.com' if self.location_id != 'global' else None)